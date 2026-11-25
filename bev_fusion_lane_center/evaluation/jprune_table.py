"""CPU side of the E345 probe (D99 step 0): scoring of the point folders, the paired scene bootstrap, the halves, the classes, the greedy curves, the plan of the scored points, the controls and the table writer. The passes themselves are `jprune_probe.py`.

    wsl bash tools/env/_py14.sh jprune_table.py plan  --sweep runs/kept/e345/e312_p_s0/sweep.json --out runs/kept/e345/e312_p_s0/score_points.json [--runs-dir runs]      # section 3: the point sets scored in full (+ v once the handle-scale points are scored), from the sweep of SEED 0
    wsl bash tools/env/_py14.sh jprune_table.py score --run e312_p_s0 --points runs/kept/e345/e312_p_s0/score_points.json [--only 12,101]                           # the stored scorer (`jends_eval.py score`) on the point folders of `runs/`, tag e345<NNN>
    wsl bash tools/env/_py14.sh jprune_table.py boot  --run e312_p_s0 --points runs/kept/e345/e312_p_s0/score_points.json [--only 12,101] [--n 1000]              # the paired scene bootstrap of every scored point against the null folder on the judged read-outs
    wsl bash tools/env/_py14.sh jprune_table.py table --sweeps runs/kept/e345/e312_p_s0,runs/kept/e345/e312_p_s1,runs/kept/e345/e324_s_s0,runs/kept/e345/e324_s_s1 --out experiments/E345_pruning-sensitivity-step0 [--runs-dir runs]
    (the tests are tests/test_jprune_probe.py)

Job flow (one model, seeds 0 and 1; relative paths are relative to the problem folder): (1) GPU: `jprune_probe.py sweep` on both seeds (null twice, controls, the 84 points, random / reverse draws); (2) CPU: `plan` on the SEED-0 sweep -> `score_points.json` (18 point sets and the second null 900, without v);
(3) GPU: `sweep --score-points <that file>` again on both seeds (only the dumps are new; seed 1 may take seed 0's file; `calib.pt` keeps the channel sets); (4) CPU: `score` of the dumps, first the handle-scale points and the nulls (`--only`); (5) CPU: `plan` again (now with v, id 120); (6) GPU: `sweep --score-points` again (the dump of v);
(7) CPU: `score` of the rest; (8) CPU: `boot`; (9) `table`. `score --points` and `boot --points` read the family (E312 / E324 s) from the `sweep.json` next to the points file, so a seed-1 run may point at seed 0's `score_points.json` (the sweep.json must lie next to it: copy both, or give the seed-1 folder its own copy).
The scorer writes into `runs/` of this checkout (`jends_eval.py score` has that place built in), so `score` and `boot` have no `--runs-dir`.

Loss L is always the null's minus the ablated point's number, positive when worse (for dup_frac, where lower is better: ablated minus null). The two halves: the 22 VAL scenes in sorted order, even index = SELECTION (decides classes, fractions, the order of the curves), odd = JUDGEMENT (judges v and the
sizes); the full-VAL number is printed next to them. Classes (card section 3): a point scored in full: FREE L <= 0.005, CHEAP <= 0.02, COSTLY <= 0.05, else CRITICAL, L = topo F1 loss on the WORSE read-out and the worse seed; a point only swept: ends_mean6 with 0.003 / 0.01 / 0.03, labelled "sweep".
Bootstrap (`boot`): `jdiag.boot` (imported, not edited; 1000 resamples of the VAL scenes, seed 0, it only prints, so its lines are parsed: 4 decimals) is called with A = the ablated point and B = the null; jdiag's sign is A - B, so for the topo F1, junction F1 and the other higher-is-better numbers the
LOSS (null minus ablated) is the negative, with the interval ends swapped; for dup_frac A - B is the loss. The intervals go to `<card stem>.boot.json` next to the point's card. They are of the full-VAL loss (jdiag.boot resamples all 22 scenes; there is no half-wise interval): a point whose point-estimate class (selection half) is FREE keeps it only
when the upper end of the full-VAL loss interval is <= 0.01 on every judged read-out and seed (card section 4), else it is CHEAP; without an interval the point-estimate class stands and the label says so.
Sweep-FREE (a unit): ends_mean6, pair AP and the keypoint position error within 0.003 / 0.003 / 0.05 m, all three. Where the card is silent: the unit's FREE fraction (and v) uses the three-criterion sweep-FREE and takes the largest ladder fraction whose point is sweep-FREE; the random control compares the ranked loss with the MEAN of the three draws; v takes the
handle-scale fraction from the full class on SEED 0 (the plan runs before seed 1 is scored); the four "largest saved GMACs" slots of the plan count only the units of the two HEADS (the pyramid is a junction view that step 1 cannot deliver); an identical point set named twice (a top-4 unit that is also a tercile point) is scored once, so the plan can hold fewer than 19 sets; the second null (id 900) is scored as
well, for the spread control; the knee S_c needs the loss per GMAC to rise above twice the previous slope AND above 0.01 per GMAC (the 0.01 band), so noise does not make a knee; the "pyramid view" prefixes are cut on the MACs saved in the pyramid's own layers (4.677 GMACs), the heads-only ones on the MACs saved in the two heads (3.278);
a null point (0, 900) is scored with every read-out of the model (E324 s: snapped, PRIMARY, snap-nogeom) for the stored-card check, every other point with the two judged ones.
"""
import argparse
import contextlib
import csv
import io
import json
import os
import pickle
import re
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", "tools/viz", ".."):
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
RULE = (("topo_F1", 0.01), ("dup_frac", 0.01), ("junction_F1", 0.03))                                                              # the card's `rule` clauses: max loss of v on every model-seed
HEADS_TARGETS, PYR_TARGETS = (0.10, 0.20, 0.35, 0.50), (0.10, 0.25)
SMALL_GMACS = 0.01
KNEE_FLOOR = 0.01                                                                                                                    # the 0.01 noise band per GMAC: a rise of the loss per GMAC below it is no knee
FREE_UPPER = 0.01                                                                                                                   # a FREE class needs the upper end of the loss interval at or below this (card section 4)
BOOT_LINE = re.compile(r"^\s*(\S+)\s+A - B: point (\S+), bootstrap mean (\S+), 95 % \[(\S+), (\S+)\]")
NULL2 = 900                                                                                                                         # the folder number of the second null pass


def classify(loss, limits):
    return NAMES[sum(loss > l for l in limits)]


def seed_of(run):
    return int(run.rsplit("_s", 1)[1])


def point_run(run, pid):
    return f"e345_p{pid:03d}_{run}"


def card_stem(pid, opts):
    import qends_eval as QE
    return QE.card_name(SimpleNamespace(thr=THR, tag=f"e345{pid:03d}", **opts))[:-5]


# ---------------------------------------------------------------- scoring and bootstrap
def _paths(runs_dir, run, pid, opts):
    d = os.path.join(runs_dir, point_run(run, pid))
    stem = card_stem(pid, opts)
    return d, stem, os.path.join(d, stem + ".json"), os.path.join(d, stem + ".counts.pkl")


def score_point(run, family, pid, runs_dir, readouts=None):
    """the stored scorer on the point folder `runs/e345_p<NNN>_<run>/` (tag e345<NNN>, thr 0.1): a null point (0, 900) with every read-out of the model (the stored-card check needs them all), any other point with the two judged ones. A read-out that has its counts file is skipped
    (a tagged card is never overwritten); a folder without `point.json` is an unfinished dump and is refused; a card json without its counts file (an earlier scoring crashed between the two writes) is reported and left for you to move away"""
    import jends_eval as JEV
    if readouts is None:
        readouts = tuple(READOUTS[family]) if pid in (0, NULL2) else JUDGED[family]
    assert os.path.exists(os.path.join(runs_dir, point_run(run, pid), "point.json")), f"{os.path.join(runs_dir, point_run(run, pid))} has no point.json: the dump is not complete (rerun the sweep with --score-points)"
    for name in readouts:
        opts = READOUTS[family][name]
        d, stem, card, counts = _paths(runs_dir, run, pid, opts)
        if os.path.exists(counts):
            continue
        if os.path.exists(card):
            raise RuntimeError(f"{card} exists without {counts}: an earlier scoring crashed between the two writes (`qends_eval.cmd_score` writes the card first and never overwrites a tagged card). Move or delete that card json by hand, then score again")
        JEV.score(SimpleNamespace(run=point_run(run, pid), seed=seed_of(run), thr=THR, tag=f"e345{pid:03d}", **opts))


def boot_point(run, family, pid, runs_dir, n=1000, readouts=None):
    """the paired scene bootstrap of a scored point against the null folder on the judged read-outs: `jdiag.boot` with A = the ablated point, B = the null (it prints; its lines are parsed). Written as LOSS (null minus ablated; dup_frac: ablated minus null, so jdiag's A - B is
    negated except for dup_frac and the interval ends swap) to `<card stem>.boot.json` next to the point's card; an existing file is kept"""
    import jdiag
    for name in readouts or JUDGED[family]:
        opts = READOUTS[family][name]
        d, stem, _, counts = _paths(runs_dir, run, pid, opts)
        d0, stem0, _, counts0 = _paths(runs_dir, run, 0, opts)
        out = os.path.join(d, stem + ".boot.json")
        if os.path.exists(out):
            continue
        assert os.path.exists(counts) and os.path.exists(counts0), f"{counts} / {counts0}: score the point and the null first"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            jdiag.boot(d, stem, d0, stem0, n=n)                                        # an absolute run folder: `os.path.join(ROOT, "runs", <absolute>)` is the folder itself
        parsed = {m.group(1): tuple(float(m.group(i)) for i in range(2, 6)) for m in map(BOOT_LINE.match, buf.getvalue().splitlines()) if m}
        assert {"topo_F1", "junction_F1", "dup_frac"} <= set(parsed), f"jdiag.boot printed no interval lines:\n{buf.getvalue()}"
        loss = {k: (dict(point=pt, mean=mn, lo=lo, hi=hi) if k == "dup_frac" else dict(point=-pt, mean=-mn, lo=-hi, hi=-lo)) for k, (pt, mn, lo, hi) in parsed.items()}
        PP.atomic_write(out, json.dumps(dict(point=pid, readout=name, resamples=n, scenes=22, a=f"{os.path.basename(d)}/{stem}", b=f"{os.path.basename(d0)}/{stem0}", note="loss = null minus ablated (dup_frac: ablated minus null): the sign of jdiag.boot's A - B negated, ends swapped; full-VAL loss, 4 decimals", loss=loss), indent=1).encode())


# ---------------------------------------------------------------- one model-seed
class RunData:
    _cache = {}                                                                         # the counts files are read once (a scored card is 5 MB, the table asks for each several times)

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
        """-> dict(m6, ap, pos, apos): ends_mean6, pair AP, mean position error of the matched keypoints (m) and of the attached ends, of a pass over the frames of the half"""
        r, m = self.rec["passes"][key], self.mask(half)
        c, k = np.array(r["counts"])[m].sum(0), np.array(r["kp"])[m].sum(0)
        return dict(m6=float(np.mean(PP.six(c))), ap=r["pair"][half]["ap"], pos=float(k[1] / max(k[0], 1)), apos=float(k[3] / max(k[0], 1)))

    def delta(self, key, half):
        """loss of a pass against the null pass: d6 and dap positive when worse, dpos the increase of the larger position error (keypoint, attached end)"""
        a, n = self.sweep(key, half), self.sweep("null", half)
        return dict(d6=n["m6"] - a["m6"], dap=n["ap"] - a["ap"], dpos=max(a["pos"] - n["pos"], a["apos"] - n["apos"]))

    def counts(self, pid, readout):
        p = os.path.join(self.runs_dir, point_run(self.run, pid), card_stem(pid, READOUTS[self.family][readout]) + ".counts.pkl")
        if p not in self._cache:
            self._cache[p] = pickle.load(open(p, "rb")) if os.path.exists(p) else None
        return self._cache[p]

    def full(self, pid, readout, half, key="topo_F1"):
        """a number of the scored graph on the frames of the half (None when the point is not scored), pooled from the per-frame counts"""
        import qgraph_eval as QE
        c = self.counts(pid, readout)
        if c is None:
            return None
        return QE.graph_numbers([c["head"][i] for i in np.where(self.mask(half))[0]])[key]

    def full_loss_by_readout(self, pid, half, key="topo_F1", readouts=None):
        """-> {read-out: loss of the scored point (null minus the point; dup_frac, where lower is better: the point minus the null)}, None for a read-out that is not scored"""
        sign = -1.0 if key == "dup_frac" else 1.0
        out = {}
        for r in readouts or JUDGED[self.family]:
            a, b = self.full(0, r, half, key), self.full(pid, r, half, key)
            out[r] = None if a is None or b is None else sign * (a - b)
        return out

    def full_loss(self, pid, half, key="topo_F1", readouts=None):
        """loss of the scored point on the WORSE read-out (None when a read-out is not scored)"""
        ls = list(self.full_loss_by_readout(pid, half, key, readouts).values())
        return None if any(x is None for x in ls) else max(ls)

    def boot(self, pid, readout):
        """the bootstrap file of a point and read-out (`boot_point`), None when there is none"""
        p = os.path.join(self.runs_dir, point_run(self.run, pid), card_stem(pid, READOUTS[self.family][readout]) + ".boot.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def loss_upper(self, pid, key="topo_F1"):
        """the upper end of the 95 % interval of the full-VAL loss on the WORSE judged read-out (None when one has no interval)"""
        b = [self.boot(pid, r) for r in JUDGED[self.family]]
        return None if any(x is None for x in b) else max(x["loss"][key]["hi"] for x in b)


def sweep_free(d):
    return d["d6"] <= BANDS[0] and d["dap"] <= BANDS[1] and d["dpos"] <= BANDS[2]


class Model:
    """both seeds of a model: classes (worse seed), FREE fractions, the curves"""

    def __init__(self, runs, units, cost):
        self.runs, self.units, self.cost = runs, units, cost                           # units: jprune_probe.Unit list, cost: jprune_cost.Cost (CPU)
        self.by = {u.name: u for u in units}
        self.pid = {(p["unit"], p["f"]): p["id"] for p in runs[0].rec["points"]}

    def klass(self, pid, half="sel"):
        """(class, label, loss) of a single point: from the full scoring when every seed has it, else from the sweep (ends_mean6); the worse seed. A FREE point estimate holds only when the upper end of the full-VAL loss interval (`boot`) is <= 0.01 on every read-out and seed
        (label "full + interval"; without an interval the label says "point estimate only")"""
        full = [r.full_loss(pid, half) for r in self.runs]
        if all(x is not None for x in full):
            cls, loss = classify(max(full), FULL_LIMITS), max(full)
            if cls != "FREE":
                return cls, "full", loss
            ups = [r.loss_upper(pid) for r in self.runs]
            if any(u is None for u in ups):
                return cls, "full, point estimate only", loss
            return ("FREE" if max(ups) <= FREE_UPPER else "CHEAP"), "full + interval", loss
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
                out.append(dict(unit=u.name, f=f, dloss=cur["d6"] - prev["d6"], dmacs=(cur["macs"] - prev["macs"]) / 1e9, dparams=cur["params"] - prev["params"]))
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
        """the prefixes of the curve closest to the targets (shares of `total` saved MACs; `key` "heads": the MACs saved in the two heads, "macs_pyr": in the pyramid's layers) -> [(target, {unit: f}, saved dict)]; the end of the curve when the units run out before a target"""
        sets, cum = [], {}
        for i in order:
            cum = dict(cum, **{i["unit"]: i["f"]})
            sets.append((dict(cum), self.saved(cum)))
        val = lambda s: s["macs_pyr"] if key == "macs_pyr" else s["macs_maps"] + s["macs_pair"]  # noqa: E731
        return [(t, *min(sets, key=lambda x: abs(val(x[1]) - t * total))) for t in targets]


def sizes(prefix_loss):
    """S_a / S_b / S_c from the scored prefixes [(saved GMACs, L judgement half, worse read-out, worse seed)] (ascending): the largest with L <= 0.01, <= 0.03, and the knee (the prefix after which the loss per saved GMAC more than doubles the previous slope, floored at 0.01 per GMAC)"""
    p = sorted(prefix_loss)
    a = max([x for x in p if x[1] <= 0.01], default=None, key=lambda x: x[0])
    b = max([x for x in p if x[1] <= 0.03], default=None, key=lambda x: x[0])
    slope = [(y[1] - x[1]) / max(y[0] - x[0], 1e-9) for x, y in zip(p, p[1:])]
    c = next((p[i] for i in range(1, len(slope)) if slope[i] > 2 * max(slope[i - 1], KNEE_FLOOR)), None)                  # the previous slope is floored at the 0.01 band per GMAC: a rise of noise is no knee
    return dict(S_a=a, S_b=b, S_c=c)


def plan(model, handle=("je.scale.1",)):
    """the point sets scored in full (card section 3, chosen on seed 0): null, the 4 units of the two HEADS (not the pyramid: a junction view step 1 cannot deliver) with the largest saved GMACs among those with a sweep-FREE fraction, 3 stratified single points (terciles of the sweep loss, seed 345), the handle-scale points, the paint point,
    the prefixes of both curves (S1 - S4 heads-only, P1 - P2 pyramid view), v once the handle-scale points are scored in full, and the second null (id 900). -> [{id, name, set}] without duplicate sets"""
    r0 = model.runs[0]
    P = {}                                                                                                            # key (frozen set) -> entry, in order

    def add(pid, name, spec):
        P.setdefault(tuple(sorted(spec.items())), dict(id=pid, name=name, set=spec))
    add(0, "null", {})
    free = [(model.saved({u.name: model.free_fraction(u.name, [r0])})["macs"], u.name) for u in model.units if u.group in ("maps", "pair") and model.free_fraction(u.name, [r0]) > 0]
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
    pids = [model.pid[(n, f)] for n in handle for f in model.by[n].fracs]
    if all(r0.full_loss(i, "sel") is not None for i in pids):
        v = {}
        for u in model.units:
            if u.group in ("maps", "pair"):
                f = max([f for f in u.fracs if r0.full_loss(model.pid[(u.name, f)], "sel") <= FULL_LIMITS[0]], default=0.0) if u.name in handle else model.free_fraction(u.name, [r0])
                if f > 0:
                    v[u.name] = f
        add(120, "v (all-FREE heads)", v)
    return list(P.values()) + [dict(id=NULL2, name="null (second pass)", set={})]


# ---------------------------------------------------------------- controls
def spearman(a, b):
    """rank correlation with the average rank for ties (nan for fewer than 3 values or a constant input)"""
    from scipy.stats import rankdata
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def control_report(run):
    """the random / reverse / proxy controls of one model-seed (card section 2, kill criterion): the shares of the non-tied ladder units where the ranked removal at f = 0.5 loses less than the random one (mean of the 3 draws) and the reverse one loses more, whole VAL; the probe's null spread and
    plumbing numbers; the proxy check (Spearman of the sweep loss against the full topo F1 loss over the single points scored in full, whole VAL, worse read-out)"""
    rank = rev = tied = n = 0
    for u in (u for u, pts in run.units.items() if len(pts) > 1):
        p = next(p for p in run.units[u] if p["f"] == 0.5)
        a = run.delta(f"p{p['id']:03d}", "all")["d6"]
        rd = float(np.mean([run.delta(f"rand{s}_{u}", "all")["d6"] for s in range(3)]))
        rv = run.delta(f"rev_{u}", "all")["d6"]
        if abs(a) < 1e-6 and abs(rd) < 1e-6:
            tied += 1
            continue
        n += 1
        rank += a < rd
        rev += rv > a
    both = [(run.delta(f"p{p['id']:03d}", "all")["d6"], run.full_loss(p["id"], "all")) for p in run.rec["points"] if run.full_loss(p["id"], "all") is not None]
    return dict(units=n, tied=tied, ranked_beats_random=rank / max(n, 1), reverse_worse=rev / max(n, 1), null_spread=run.rec["controls"].get("null_spread_ends_mean6"), plumbing=run.rec["controls"].get("plumbing"),
                proxy_spearman=spearman([x[0] for x in both], [x[1] for x in both]) if len(both) > 2 else None, proxy_points=len(both))


def null_check(run, other=0):
    """the scored null against the stored cards (`other` = 0) or against the second null (`other` = NULL2): the worst difference per read-out in units of the tolerance (topo F1 2e-4, any other real number 3e-3; <= 1 passes), None when not scored"""
    out = {}
    for name, stored in STORED[run.family].items():
        cur = os.path.join(run.runs_dir, point_run(run.run, 0), card_stem(0, READOUTS[run.family][name]) + ".json")
        ref = os.path.join(run.runs_dir, run.run, stored) if other == 0 else os.path.join(run.runs_dir, point_run(run.run, other), card_stem(other, READOUTS[run.family][name]) + ".json")
        if not (os.path.exists(cur) and os.path.exists(ref)):
            out[name] = None
            continue
        a, b = (json.load(open(x))["numbers"]["head"] for x in (ref, cur))
        out[name] = max(abs(a[k] - b[k]) / (NULL_TOL["topo_F1"] if k == "topo_F1" else NULL_TOL["other"]) for k in a if k in b and isinstance(a[k], float))
    return out


# ---------------------------------------------------------------- the table
def sets_report(model, entries):
    """the scored point sets of the plan (prefixes S / P, v, ...): saved GMACs and the loss per half on the worse read-out, per seed (None while not scored), the rule clauses of the card for v, the sizes S_a / S_b / S_c from the heads-only prefixes"""
    rows, pref = [], []
    for e in (e for e in entries if e["id"] >= 100 and e["id"] != NULL2):
        sv = model.saved(e["set"])
        row = dict(id=e["id"], name=e["name"], units=len(e["set"]), saved_gmacs=(sv["macs_pyr"] if e["name"].startswith("P") else sv["macs_maps"] + sv["macs_pair"]) / 1e9, saved_params=sv["params"])
        for r in model.runs:
            for h in ("sel", "jud", "all"):
                row[f"s{seed_of(r.run)}_L_{h}"] = r.full_loss(e["id"], h)
            row[f"s{seed_of(r.run)}_L_hi"] = r.loss_upper(e["id"])
        if e["id"] == 120:
            row["rule_holds"] = {f"s{seed_of(r.run)}": [None if r.full_loss(e["id"], "jud", k) is None else r.full_loss(e["id"], "jud", k) <= m for k, m in RULE] for r in model.runs}
        if e["name"].startswith("S"):
            ls = [r.full_loss(e["id"], "jud") for r in model.runs]
            if all(x is not None for x in ls):
                pref.append((row["saved_gmacs"], max(ls)))
        rows.append(row)
    return rows, sizes(pref) if pref else None


def write_table(models, out_dir, plans=None):
    """results.md and table.csv of E345 from the sweeps (and the scored cards that exist)"""
    rows, md = [], ["# E345 results (written by jprune_table.py table from the sweeps; VAL, the selection half decides, the judgement half judges)", ""]
    for name, m in models.items():
        md += [f"## {name}", "", "null against the stored cards (units of the tolerance, <= 1 passes), seed 0 / seed 1: " + json.dumps(null_check(m.runs[0])) + " / " + json.dumps(null_check(m.runs[1])),
               "the two nulls against each other: " + json.dumps(null_check(m.runs[0], NULL2)) + " / " + json.dumps(null_check(m.runs[1], NULL2)), ""]
        md += [f"controls {r.run}: " + json.dumps(control_report(r)) for r in m.runs] + [""]
        if plans and name in plans:
            sr, sz = sets_report(m, plans[name])
            md += ["| set | units | saved GMACs | L sel s0 | L jud s0 | 95 % upper s0 | L sel s1 | L jud s1 | 95 % upper s1 |", "|---|---|---|---|---|---|---|---|---|"]
            fmt = lambda x: "-" if x is None else f"{x:+.4f}"  # noqa: E731
            md += [f"| {r['name']} | {r['units']} | {r['saved_gmacs']:.3f} | {fmt(r.get('s0_L_sel'))} | {fmt(r.get('s0_L_jud'))} | {fmt(r.get('s0_L_hi'))} | {fmt(r.get('s1_L_sel'))} | {fmt(r.get('s1_L_jud'))} | {fmt(r.get('s1_L_hi'))} |" for r in sr] + ["", f"sizes (judgement half, worse seed and read-out): {json.dumps(sz)}", ""]
        ro = JUDGED[m.runs[0].family]
        md += [f"| unit | f | saved GMACs | saved Mparams | sweep L s0 (sel) | sweep L s1 (sel) | full L s0 sel ({' / '.join(ro)}) | full L s1 sel ({' / '.join(ro)}) | class (worse seed, sel) | label | sweep L judgement |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda x: "-" if x is None else f"{x:+.4f}"  # noqa: E731
        for p in m.runs[0].rec["points"]:
            key = f"p{p['id']:03d}"
            cl, lab, loss = m.klass(p["id"])
            d = [r.delta(key, "sel")["d6"] for r in m.runs]
            sv = m.runs[0].rec["passes"][key]
            by = [r.full_loss_by_readout(p["id"], "sel") for r in m.runs]
            md.append(f"| {p['unit']} | {p['f']} | {sv['saved_macs'] / 1e9:.4f} | {sv['saved_params'] / 1e6:.4f} | {d[0]:+.4f} | {d[1]:+.4f} | " + " | ".join(" / ".join(fmt(b[x]) for x in ro) for b in by) + f" | {cl} | {lab} | {max(r.delta(key, 'jud')['d6'] for r in m.runs):+.4f} |")
            rows.append(dict(model=name, id=p["id"], unit=p["unit"], f=p["f"], group=p["group"], saved_gmacs=sv["saved_macs"] / 1e9, saved_params=sv["saved_params"], klass=cl, label=lab, class_loss=loss,
                             **{f"s{seed_of(r.run)}_{k}_{h}": r.delta(key, h)[k] for r in m.runs for h in ("sel", "jud", "all") for k in ("d6", "dap", "dpos")}, **{f"s{seed_of(r.run)}_full_{h}": r.full_loss(p["id"], h) for r in m.runs for h in ("sel", "jud", "all")},
                             **{f"s{seed_of(r.run)}_full_{x}_{h}": r.full_loss_by_readout(p["id"], h)[x] for r in m.runs for x in ro for h in ("sel", "jud", "all")}, **{f"s{seed_of(r.run)}_full_hi": r.loss_upper(p["id"]) for r in m.runs}))
        md.append("")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "results.md"), "w").write("\n".join(md) + "\n")
    first = ["model", "id", "unit", "f", "group", "saved_gmacs", "saved_params", "klass", "label", "class_loss"]
    keys = first + sorted({k for r in rows for k in r} - set(first))
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
    for name in ("score", "boot"):
        s = sp.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--points", required=True, help="score_points.json; the sweep.json next to it says which family (E312 / E324 s) the run is")
        s.add_argument("--only", default=None, help="comma separated point ids")
        if name == "boot":
            s.add_argument("--n", type=int, default=1000, help="resamples of the VAL scenes")
    s = sp.add_parser("table")
    s.add_argument("--sweeps", required=True, help="comma separated sweep folders, the two seeds of one model next to each other (e312_p_s0,e312_p_s1,e324_s_s0,e324_s_s1)")
    s.add_argument("--out", required=True)
    s.add_argument("--runs-dir", default=None)
    a = ap.parse_args()
    for k in ("out", "runs_dir", "points", "sweep"):
        if hasattr(a, k):
            setattr(a, k, PP.rootpath(getattr(a, k)))
    if hasattr(a, "sweeps"):
        a.sweeps = ",".join(PP.rootpath(x) for x in a.sweeps.split(","))
    if a.cmd == "plan":
        d = os.path.dirname(a.sweep) or "."
        pl = plan(load_model([d, d], a.runs_dir))
        json.dump(pl, open(a.out, "w"), indent=1)
        print(f"{len(pl)} point sets -> {a.out}" + ("" if any(p["id"] == 120 for p in pl) else " (v needs the handle-scale points scored in full: run `score --only <their ids>`, then `plan` again)"))
    elif a.cmd in ("score", "boot"):
        rd = os.path.join(ROOT, "runs")
        fam = "e324" if json.load(open(os.path.join(os.path.dirname(a.points) or ".", "sweep.json")))["attach"] else "e312"
        for p in json.load(open(a.points)):
            if a.only is None or p["id"] in {int(x) for x in a.only.split(",")}:
                if a.cmd == "score":
                    score_point(a.run, fam, p["id"], rd)
                elif p["id"] not in (0, NULL2):
                    boot_point(a.run, fam, p["id"], rd, a.n)
    else:
        d = a.sweeps.split(",")
        ms = {os.path.basename(d[2 * i].rstrip("/")).rsplit("_s", 1)[0]: load_model(d[2 * i:2 * i + 2], a.runs_dir) for i in range(len(d) // 2)}
        plans = {n: json.load(open(os.path.join(d[2 * i], "score_points.json"))) for i, n in enumerate(ms) if os.path.exists(os.path.join(d[2 * i], "score_points.json"))}
        write_table(ms, a.out, plans)


if __name__ == "__main__":
    main()
