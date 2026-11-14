"""CPU side of the E345 probe (D99 step 0): scoring of the point folders, the halves, the classes, the greedy curves, the plan of the scored points and the table writer. The passes themselves are `jprune_probe.py`.

    wsl bash tools/env/_py14.sh jprune_table.py plan  --sweep temp/e345/e312_p_s0/sweep.json --out temp/e345/e312_p_s0/score_points.json [--runs-dir runs]     # section 3: the points scored in full (+ v once the handle-scale points are scored), from the sweep of SEED 0
    wsl bash tools/env/_py14.sh jprune_table.py score --run e312_p_s0 --points temp/e345/e312_p_s0/score_points.json [--runs-dir runs] [--only 12,101]            # the stored scorer (`jends_eval.py score`) on the point folders, tag e345<NNN>, the read-outs of the model
    wsl bash tools/env/_py14.sh jprune_table.py table --sweeps temp/e345/e312_p_s0,temp/e345/e312_p_s1,temp/e345/e324_s_s0,temp/e345/e324_s_s1 --out experiments/E345_pruning-sensitivity-step0
    wsl bash tools/env/_py14.sh jprune_table.py selftest

Loss L is always the null's minus the ablated point's number, positive when worse. The two halves: the 22 VAL scenes in sorted order, even index = SELECTION (decides classes, fractions, the order of the curves), odd = JUDGEMENT (judges v and the sizes); the full-VAL number is printed next to them.
Classes (card section 3): a point scored in full: FREE L <= 0.005, CHEAP <= 0.02, COSTLY <= 0.05, else CRITICAL, L = topo F1 loss on the WORSE read-out and the worse seed; a point only swept: ends_mean6 with 0.003 / 0.01 / 0.03, labelled "sweep".
Sweep-FREE (a unit): ends_mean6, pair AP and the keypoint position error within 0.003 / 0.003 / 0.05 m. Where the card is silent: the unit's FREE fraction is the largest ladder fraction whose point is sweep-FREE; the random control compares the ranked loss with the MEAN of the three draws; v takes the handle-scale fraction
from the full class on SEED 0 (the plan runs before seed 1 is scored); an identical point set named twice (a top-4 unit that is also a tercile point) is scored once, so the plan can hold fewer than 19 sets.
"""
import argparse
import csv
import json
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "training/grid", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jprune_probe as PP  # noqa: E402
from jprune_cost import Cost  # noqa: E402

THR = 0.1
READOUTS = {"e312": {"snap": dict(snap=True), "unsnap": {}},                                                                         # the stored cards of E312: snapped (the rule's) and unsnapped (differentiable)
            "e324": {"snap": dict(snap=True), "primary": dict(snap=True, snap_nogeom=True, attach=True), "snap_nogeom": dict(snap=True, snap_nogeom=True)}}    # E324 s: snapped, PRIMARY (attached ends + snap-nogeom), snap-nogeom (null only)
JUDGED = {"e312": ("snap", "unsnap"), "e324": ("snap", "primary")}                                                                   # the two read-outs of every point but the null
STORED = {"e312": {"snap": "ends_card_snap_thr0.1.json", "unsnap": "ends_card_thr0.1.json"},
          "e324": {"snap": "ends_card_snap_thr0.1.json", "primary": "ends_card_snap_nogeom_attach_thr0.1.json", "snap_nogeom": "ends_card_snap_nogeom_thr0.1.json"}}
FULL_LIMITS, SWEEP_LIMITS = (0.005, 0.02, 0.05), (0.003, 0.01, 0.03)
NAMES = ("FREE", "CHEAP", "COSTLY", "CRITICAL")
BANDS = (0.003, 0.003, 0.05)                                                                                                        # sweep-FREE: ends_mean6, pair AP, position error (m)
NULL_TOL = dict(topo_F1=2e-4, other=3e-3)
HEADS_TARGETS, PYR_TARGETS = (0.10, 0.20, 0.35, 0.50), (0.10, 0.25)
SMALL_GMACS = 0.01


def classify(loss, limits):
    return NAMES[sum(loss > l for l in limits)]


def seed_of(run):
    return int(run.rsplit("_s", 1)[1])


def point_run(run, pid):
    return f"e345_p{pid:03d}_{run}"


def card_stem(pid, opts):
    import qends_eval as QE
    return QE.card_name(SimpleNamespace(thr=THR, tag=f"e345{pid:03d}", **opts))[:-5]


# ---------------------------------------------------------------- scoring
def score_point(run, family, pid, runs_dir, readouts=None):
    """the stored scorer on the point folder `runs/e345_p<NNN>_<run>/` (tag e345<NNN>, thr 0.1) for the read-outs of the model; a read-out that has its counts file is skipped (a tagged card is never overwritten)"""
    import jends_eval as JEV
    for name in readouts or READOUTS[family]:
        opts = READOUTS[family][name]
        if os.path.exists(os.path.join(runs_dir, point_run(run, pid), card_stem(pid, opts) + ".counts.pkl")):
            continue
        JEV.score(SimpleNamespace(run=point_run(run, pid), seed=seed_of(run), thr=THR, tag=f"e345{pid:03d}", **opts))


# ---------------------------------------------------------------- one model-seed
class RunData:
    def __init__(self, sweep_dir, runs_dir=None):
        self.rec = json.load(open(os.path.join(sweep_dir, "sweep.json")))
        self.run = self.rec["run"]
        self.family = "e324" if self.rec["attach"] else "e312"
        self.runs_dir = runs_dir or os.path.join(ROOT, "runs")
        self.half = PP.halves(self.rec["rows"])
        self.units = {p["unit"]: [] for p in self.rec["points"]}
        for p in self.rec["points"]:
            self.units[p["unit"]].append(p)

    def mask(self, half):
        return {"sel": self.half == 0, "jud": self.half == 1, "all": np.ones(len(self.half), bool)}[half]

    def sweep(self, key, half):
        """-> dict(m6, ap, pos, apos): ends_mean6, pair AP, mean position error of the matched keypoints (m), of the attached ends, for a pass over the frames of the half"""
        r, m = self.rec["passes"][key], self.mask(half)
        c, k = np.array(r["counts"])[m].sum(0), np.array(r["kp"])[m].sum(0)
        return dict(m6=float(np.mean(PP.six(c))), ap=r["pair"][half]["ap"], pos=float(k[1] / max(k[0], 1)), apos=float(k[3] / max(k[0], 1)))

    def delta(self, key, half):
        """loss of a pass against the null pass: d6 and dap positive when worse, dpos the increase of the larger position error (keypoint, attached end)"""
        a, n = self.sweep(key, half), self.sweep("null", half)
        return dict(d6=n["m6"] - a["m6"], dap=n["ap"] - a["ap"], dpos=max(a["pos"] - n["pos"], a["apos"] - n["apos"]))

    def counts(self, pid, readout):
        p = os.path.join(self.runs_dir, point_run(self.run, pid), card_stem(pid, READOUTS[self.family][readout]) + ".counts.pkl")
        return pickle.load(open(p, "rb")) if os.path.exists(p) else None

    def full(self, pid, readout, half, key="topo_F1"):
        """a number of the scored graph on the frames of the half (None when the point is not scored), from the per-frame counts"""
        import qgraph_eval as QE
        c = self.counts(pid, readout)
        if c is None:
            return None
        sel = np.where(self.mask(half))[0] if half != "all" else range(len(c["head"]))
        return QE.graph_numbers([c["head"][i] for i in sel])[key]

    def full_loss(self, pid, half):
        """topo F1 loss of the scored point on the WORSE read-out (None when a read-out is not scored)"""
        ls = [None if self.full(pid, r, half) is None or self.full(0, r, half) is None else self.full(0, r, half) - self.full(pid, r, half) for r in JUDGED[self.family]]
        return None if any(x is None for x in ls) else max(ls)


def sweep_free(d):
    return d["d6"] <= BANDS[0] and d["dap"] <= BANDS[1] and d["dpos"] <= BANDS[2]


class Model:
    """both seeds of a model: classes (worse seed), FREE fractions, the curves"""

    def __init__(self, runs, units, cost):
        self.runs, self.units, self.cost = runs, units, cost                           # units: jprune_probe.Unit list, cost: jprune_cost.Cost (CPU)
        self.by = {u.name: u for u in units}
        self.pid = {(p["unit"], p["f"]): p["id"] for p in runs[0].rec["points"]}

    def klass(self, pid, half="sel"):
        full = [r.full_loss(pid, half) for r in self.runs]
        if all(x is not None for x in full):
            return classify(max(full), FULL_LIMITS), "full", max(full)
        d6 = max(r.delta(f"p{pid:03d}", half)["d6"] for r in self.runs)
        return classify(d6, SWEEP_LIMITS), "sweep", d6

    def free_fraction(self, unit, runs=None, half="sel"):
        """the largest ladder fraction of a unit whose point is sweep-FREE on every given run (0.0: none)"""
        best = 0.0
        for f in self.by[unit].fracs:
            if all(sweep_free(r.delta(f"p{self.pid[(unit, f)]:03d}", half)) for r in (runs or self.runs)):
                best = f
        return best

    def saved(self, spec):
        return self.cost.saved([e for n, f in spec.items() for e in PP.count_edits(self.by[n], f, self.cost.C)])

    def increments(self, groups, run, half="sel"):
        """the (unit, f) increments of the units of the groups with their marginal loss (ends_mean6, `half` of `run`) and marginal saved MACs / parameters"""
        out = []
        for u in (u for u in self.units if u.group in groups):
            prev = dict(d6=0.0, macs=0.0, params=0.0)
            for f in u.fracs:
                key = f"p{self.pid[(u.name, f)]:03d}"
                cur = dict(d6=run.delta(key, half)["d6"], macs=run.rec["passes"][key]["saved_macs"], params=run.rec["passes"][key]["saved_params"])
                out.append(dict(unit=u.name, f=f, f_prev=0.0 if prev is None else None, dloss=cur["d6"] - prev["d6"], dmacs=(cur["macs"] - prev["macs"]) / 1e9, dparams=cur["params"] - prev["params"]))
                prev = cur
        return out

    def greedy(self, groups, run, half="sel"):
        """the order of the cumulative curve: increments by marginal loss per saved GMAC (those that save less than 0.01 GMACs after all others, by marginal loss per saved parameter), a unit's fraction only after its lower one, ties by unit name"""
        left, order = self.increments(groups, run, half), []
        while left:
            ok = [i for i in left if not any(j["unit"] == i["unit"] and j["f"] < i["f"] for j in left)]
            i = min(ok, key=lambda i: (i["dmacs"] < SMALL_GMACS, i["dloss"] / max(i["dparams"], 1e-9) if i["dmacs"] < SMALL_GMACS else i["dloss"] / max(i["dmacs"], 1e-12), i["unit"]))
            order.append(i)
            left.remove(i)
        return order

    def prefixes(self, order, targets, total, key):
        """the prefixes of the curve closest to the targets (shares of `total` saved MACs, `key`: macs_pyr / the heads' macs): -> [(target, {unit: f}, saved dict)]; the end of the curve when the units run out before a target"""
        sets, cum = [], {}
        for i in order:
            cum = dict(cum, **{i["unit"]: i["f"]})
            sets.append((dict(cum), self.saved(cum)))
        val = lambda s: s[key] if key == "macs_pyr" else s["macs_maps"] + s["macs_pair"]  # noqa: E731
        return [(t, *min(sets, key=lambda x: abs(val(x[1]) - t * total))) for t in targets]


def sizes(prefix_loss):
    """S_a / S_b / S_c from the scored prefixes [(saved GMACs, L judgement half, worse read-out, worse seed)] ascending: the largest with L <= 0.01, <= 0.03, and the knee (the prefix after which the loss per saved GMAC doubles)"""
    p = sorted(prefix_loss)
    a = max([x for x in p if x[1] <= 0.01], default=None, key=lambda x: x[0])
    b = max([x for x in p if x[1] <= 0.03], default=None, key=lambda x: x[0])
    slope = [(y[1] - x[1]) / max(y[0] - x[0], 1e-9) for x, y in zip(p, p[1:])]
    c = next((p[i] for i in range(1, len(slope)) if slope[i] > 2 * max(slope[i - 1], 1e-9)), None)
    return dict(S_a=a, S_b=b, S_c=c)


def plan(model, handle=("je.scale.1",)):
    """the point sets scored in full (card section 3, chosen on seed 0): null, the 4 units with the largest saved GMACs among those with a sweep-FREE fraction, 3 stratified single points (terciles of the sweep loss, seed 345), the handle-scale points, the paint point,
    the prefixes of both curves (S1 - S4 heads-only, P1 - P2 pyramid view), and v once the handle-scale points are scored. -> [{id, name, set}] without duplicates"""
    r0 = model.runs[0]
    P = {}                                                                                                            # key (frozen set) -> entry, in order

    def add(pid, name, spec):
        P.setdefault(tuple(sorted(spec.items())), dict(id=pid, name=name, set=spec))
    add(0, "null", {})
    free = [(model.saved({u.name: model.free_fraction(u.name, [r0])})["macs"], u.name) for u in model.units if model.free_fraction(u.name, [r0]) > 0]
    for _, n in sorted(free, reverse=True)[:4]:
        f = model.free_fraction(n, [r0])
        add(model.pid[(n, f)], f"top-saver {n}", {n: f})
    pts = sorted(r0.rec["points"], key=lambda p: r0.delta(f"p{p['id']:03d}", "sel")["d6"])
    rng = np.random.RandomState(345)
    for t in range(3):
        p = pts[t * len(pts) // 3:(t + 1) * len(pts) // 3][rng.randint(len(pts) // 3)]
        add(p["id"], f"tercile {t + 1}", {p["unit"]: p["f"]})
    for n in handle:
        for f in model.by[n].fracs:
            add(model.pid[(n, f)], f"handle {n}@{f}", {n: f})
    for u in model.units:
        if u.kind == "cols":
            add(model.pid[(u.name, 1.0)], "paint evidence", {u.name: 1.0})
    c = model.cost.all
    for k, (t, spec, _) in enumerate(model.prefixes(model.greedy(("maps", "pair"), r0), HEADS_TARGETS, c["macs_maps"] + c["macs_pair"], "heads"), 1):
        add(100 + k, f"S{k} (heads-only, {int(t * 100)} %)", spec)
    for k, (t, spec, _) in enumerate(model.prefixes(model.greedy(("pyr",), r0), PYR_TARGETS, c["macs_pyr"], "macs_pyr"), 1):
        add(110 + k, f"P{k} (pyramid view, {int(t * 100)} %)", spec)
    if all(r0.full_loss(model.pid[(n, f)], "sel") is not None for n in handle for f in model.by[n].fracs if n in model.by):
        v = {}
        for u in model.units:
            if u.group in ("maps", "pair"):
                if u.name in handle:
                    ok = [f for f in u.fracs if (r0.full_loss(model.pid[(u.name, f)], "sel") or 1.0) <= FULL_LIMITS[0]]
                    f = max(ok, default=0.0)
                else:
                    f = model.free_fraction(u.name, [r0])
                if f > 0:
                    v[u.name] = f
        add(120, "v (all-FREE heads)", v)
    return list(P.values())


# ---------------------------------------------------------------- controls
def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x)) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def control_report(run):
    """the random / reverse / proxy controls of one model-seed (card section 2, kill criterion): the shares of the non-tied ladder units where the ranked removal at f = 0.5 loses less than the random one (mean of the 3 draws) and less than the reverse one, whole VAL"""
    rec, r = run.rec, run
    rank = rand = rev = tied = 0
    n = 0
    for u in (u for u, pts in run.units.items() if len(pts) > 1):
        p = next(p for p in run.units[u] if p["f"] == 0.5)
        a = r.delta(f"p{p['id']:03d}", "all")["d6"]
        rd = float(np.mean([r.delta(f"rand{s}_{u}", "all")["d6"] for s in range(3)]))
        rv = r.delta(f"rev_{u}", "all")["d6"]
        if abs(a) < 1e-6 and abs(rd) < 1e-6:
            tied += 1
            continue
        n += 1
        rank += a < rd
        rev += rv > a
    return dict(units=n, tied=tied, ranked_beats_random=rank / max(n, 1), reverse_worse=rev / max(n, 1), null_spread=rec["controls"].get("null_spread_ends_mean6"), plumbing=rec["controls"].get("plumbing"))


def null_check(run):
    """the scored null against the stored cards (topo F1 2e-4, any other real number 3e-3): -> {read-out: worst difference} (None when the null is not scored)"""
    out = {}
    for name, stored in STORED[run.family].items():
        sp = os.path.join(run.runs_dir, run.run, stored)
        p = os.path.join(run.runs_dir, point_run(run.run, 0), card_stem(0, READOUTS[run.family][name]) + ".json")
        if not (os.path.exists(sp) and os.path.exists(p)):
            out[name] = None
            continue
        a, b = (json.load(open(x))["numbers"]["head"] for x in (sp, p))
        out[name] = max(abs(a[k] - b[k]) / (NULL_TOL["topo_F1"] if k == "topo_F1" else NULL_TOL["other"]) for k in a if k in b and isinstance(a[k], float))      # in units of the tolerance: <= 1 passes
    return out


# ---------------------------------------------------------------- the table
def write_table(models, out_dir):
    """results.md and table.csv of E345 from the sweeps (and the scored cards that exist)"""
    rows, md = [], ["# E345 results (written by jprune_table.py table from the sweeps; VAL, selection half decides, judgement half judges)", ""]
    for name, m in models.items():
        md += [f"## {name}", "", "null check against the stored cards (in units of the tolerance, <= 1 passes): " + json.dumps(null_check(m.runs[0])) + " / seed 1: " + json.dumps(null_check(m.runs[1])), ""]
        for r in m.runs:
            md += [f"controls {r.run}: " + json.dumps(control_report(r)), ""]
        md += ["| unit | f | saved GMACs | saved Mparams | sweep L s0 (sel) | sweep L s1 (sel) | class (worse seed, sel) | label | L judgement |", "|---|---|---|---|---|---|---|---|---|"]
        for p in m.runs[0].rec["points"]:
            key = f"p{p['id']:03d}"
            cl, lab, loss = m.klass(p["id"])
            d = [r.delta(key, "sel")["d6"] for r in m.runs]
            sv = m.runs[0].rec["passes"][key]
            jud = max(r.delta(key, "jud")["d6"] for r in m.runs)
            md.append(f"| {p['unit']} | {p['f']} | {sv['saved_macs'] / 1e9:.4f} | {sv['saved_params'] / 1e6:.4f} | {d[0]:+.4f} | {d[1]:+.4f} | {cl} | {lab} | {jud:+.4f} |")
            rows.append(dict(model=name, id=p["id"], unit=p["unit"], f=p["f"], group=p["group"], saved_gmacs=sv["saved_macs"] / 1e9, saved_params=sv["saved_params"], class_=cl, label=lab, class_loss=loss,
                             **{f"s{r.run[-1]}_{k}_{h}": r.delta(key, h)[k] for r in m.runs for h in ("sel", "jud", "all") for k in ("d6", "dap", "dpos")}))
        md.append("")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(md) + "\n")
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "id", "unit", "f"), k))
    with open(os.path.join(out_dir, "table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def load_model(sweeps, runs_dir=None):
    import jends_eval as JEV
    runs = [RunData(d, runs_dir) for d in sweeps]
    mdl, _ = JEV.load_model(runs[0].run, "cpu")
    return Model(runs, PP.build_units(mdl), Cost(mdl))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("plan")
    s.add_argument("--sweep", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--runs-dir", default=None)
    s = sp.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--points", required=True)
    s.add_argument("--runs-dir", default=None)
    s.add_argument("--only", default=None)
    s = sp.add_parser("table")
    s.add_argument("--sweeps", required=True, help="comma separated sweep folders, the seeds of one model next to each other (e312_p_s0,e312_p_s1,e324_s_s0,e324_s_s1)")
    s.add_argument("--out", required=True)
    s.add_argument("--runs-dir", default=None)
    sp.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit("the tests are tests/test_jprune_probe.py")
    if a.cmd == "plan":
        m = load_model([os.path.dirname(a.sweep), os.path.dirname(a.sweep)], a.runs_dir)
        pl = plan(m)
        json.dump(pl, open(a.out, "w"), indent=1)
        print(f"{len(pl)} point sets -> {a.out}" + ("" if any(p["id"] == 120 for p in pl) else " (v needs the handle-scale points scored: run plan again after `score`)"))
    elif a.cmd == "score":
        rd = a.runs_dir or os.path.join(ROOT, "runs")
        fam = "e324" if json.load(open(os.path.join(os.path.dirname(a.points), "sweep.json")))["attach"] else "e312"
        for p in json.load(open(a.points)):
            if a.only is None or p["id"] in {int(x) for x in a.only.split(",")}:
                score_point(a.run, fam, p["id"], rd)
    else:
        d = a.sweeps.split(",")
        write_table({os.path.basename(d[2 * i]).rsplit("_s", 1)[0]: load_model(d[2 * i:2 * i + 2], a.runs_dir) for i in range(len(d) // 2)}, a.out)


if __name__ == "__main__":
    main()
