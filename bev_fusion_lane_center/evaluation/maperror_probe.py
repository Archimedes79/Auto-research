"""E394 step 3 (card experiments/E394_*): how every map error of the catalogue (`net/prep/lane_maperror_data.py`) degrades the association of the champion's delivered tokens to the map segments and the lanes per direction read
through it. CPU. Step 0 of the card: the nulls and the headroom, so that the bars are derived from measurements.

    wsl bash tools/env/_py14.sh maperror_probe.py degrade --seed S [--split holdout|val] [--modes a,b|all] [--every N] [--heads p6zero,F1,z]   # -> runs/e394_step0/degrade_<split>_s<S>[_<tag>].json (never overwritten)
    wsl bash tools/env/_py14.sh maperror_probe.py selftest

The data are E391's (`mapassoc_probe.AssocData`: the cached B' tokens, the E390 road graph, the labels; the encoder `runs/e390_g7_s<S>` frozen); only the map draw differs: `ErrAssocData` builds the input map with
`lane_maperror_data.apply` from a SPEC (one mode at one level, on an otherwise EXACT map, so the effect of each error is read alone; the deciding mixtures are specs too). Heads (all read on the same draws): `p6zero` = the zero-shot
E390 association head (the step-1 head: D137's base) and its read-through, `F1` / `z` = E391's trained fusion head and the no-map head of the same seed (their weights `runs/e391_<arm>_s<S>`; they were trained on the
SIMPLE error mix only: how they behave under the new errors is what this step measures). Metrics over the delivered tokens: E391's (assoc_acc, nonfalse_acc, seg_acc, none_recall, false_assign, false_unassigned,
lanes_acc, route_f1) and `wrong_assign` (a token given a segment that is not its own, false and missing-segment tokens included: the 'wrong overwrite' of D135). `--every N` reads every N-th frame of the split.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("training/data", "training/data/prep", "net/prep", "net/model", "net/loss", "training/train", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jmapassoc as JM  # noqa: E402,F401
import lane_mapassoc_data as MA  # noqa: E402
import lane_maperror_data as ME  # noqa: E402
import lane_sdgraph_data as SG  # noqa: E402
import mapassoc_probe as MP  # noqa: E402
import sdgraph_probe as SP  # noqa: E402

OUT = os.path.join(ROOT, "runs", "e394_step0")
METRICS = MP.METRICS + ("wrong_assign", "assign_rate", "lanes_all")
SEED_DRAW = 394


class ErrAssocData(MP.AssocData):
    """`MP.AssocData` with the map draw of the error catalogue: `spec` = {mode: parameter} (fixed draws per frame, seeded by (394, frame)); wrong_scene borrows the map of `plan[i]` (a frame of another scene)"""

    def __init__(self, split, seed, spec, frames=None, plan=None, need_map=True):
        super().__init__(split, seed, frames=frames, level="exact", need_map=need_map)
        self.spec, self.plan = dict(spec), plan

    def __getitem__(self, j):
        i = self.frames[j]
        it = self.ds[i]
        G = self.sg["graphs"][i]
        rec = self.lab["frames"][i]
        rng = np.random.default_rng([SEED_DRAW, i])
        other = self.sg["graphs"][int(self.plan[i])] if self.spec.get("wrong_scene") else None
        Gin, T = ME.apply(G, rng, self.spec, other)
        pos, dirv = np.asarray(it["pos"], float), np.asarray(it["dirv"], float)
        out = {k: np.asarray(it[k]) for k in ("tok", "pos", "dirv", "e", "flags", "valid")}
        seg = rec["seg"].astype(int)
        present = T["present"]
        kind = np.full(len(seg), -1, int)                                    # 0 a present segment, 1 its segment missing from the map, 2 a false token
        lab = seg >= 0
        pres_lab = lab & present[np.clip(seg, 0, None)]
        kind[pres_lab] = 0
        kind[lab & ~pres_lab] = 1
        kind[seg == MA.FALSE] = 2
        assoc = np.where(kind == 0, seg, np.where(kind >= 1, -1, -2))
        lanes = np.where((kind == 0) & (G["kind"][np.clip(seg, 0, None)] == 0), G["lanes"][np.clip(seg, 0, None)], -1)
        lanes_all = np.where((kind <= 1) & (kind >= 0) & (G["kind"][np.clip(seg, 0, None)] == 0), G["lanes"][np.clip(seg, 0, None)], -1)     # the TRUE lane class of every labelled road token, whether or not its segment is in the map
        out.update(assoc=assoc.astype(np.int64), kind=kind.astype(np.int64), lanes=lanes.astype(np.int64), lanes_all=lanes_all.astype(np.int64), route=rec["on_lane"].astype(np.int64),
                   kept=np.asarray(rec.get("kept", it["valid"]), bool), frame=np.int64(i))
        if self.need_map:
            Gin = dict(Gin, prox=pos, pdir=dirv)
            f = SG.featurize(Gin, stated_rate=True)
            out["map"] = dict(x=f["x"], rel=f["rel"], pair=f["pair"], present=f["present"])
            out["pa"] = f["pa"]
            out["lanes_in"] = Gin["lanes_in"]
            out["route_in"] = Gin["route_in"]
            out["seg_kind"] = Gin["kind"]                                    # the INPUT map's kinds (extra nodes included)
        return out


def collate(items):
    b = MP.collate(items)
    b["lanes_all"] = torch.as_tensor(np.stack([it["lanes_all"] for it in items]))
    return b


class Lall:
    """the lane read-out against the TRUE lane class of every labelled road token (card E394: the planner needs the lanes per direction at that place, also when the map is wrong or lacks the segment)"""

    def __init__(self):
        self.d = {}

    def add(self, k, b, r, lp):
        m = b["kept"][r].numpy() & (b["lanes_all"][r].numpy() >= 0)
        self.d[k] = (np.asarray(lp)[m], b["lanes_all"][r].numpy()[m])

    def acc(self, fr):
        sel = [self.d[k] for k in fr if k in self.d]
        if not sel:
            return float("nan")
        p, t = (np.concatenate(x) for x in zip(*sel))
        return float((p == t).mean()) if len(t) else float("nan")


def plan_for(split, seed):
    lab = MA.load(split if split == "val" else "train", seed)
    sg = SG.load(split if split == "val" else "train")
    nodes = np.array([len(G["kind"]) for G in sg["graphs"]])
    return ME.other_plan(lab["scene"], nodes)


def extra_metrics(rows, fr):
    sel = [rows.a[k] for k in fr if k in rows.a]
    if not sel:
        return dict(wrong_assign=float("nan"), assign_rate=float("nan"))
    p, t, kd = (np.concatenate(x) for x in zip(*sel))
    return dict(wrong_assign=float(((p != -1) & (p != t)).mean()), assign_rate=float((p != -1).mean()))


def level_rows(split, seed, spec, every, heads, plan, log=print):
    """the rows of one spec: for each requested head a MP.Rows over the delivered tokens of the same draws -> ({head: Rows}, frame keys, scene)"""
    frames, scene = MP.eval_frames(split, seed)
    frames = list(range(len(scene))) if frames is None else frames
    frames = frames[:: max(1, every)]
    data = ErrAssocData(split if split == "val" else "train", seed, spec, frames=frames, plan=plan)
    dl = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)
    enc, heads6 = SP.load_models("g7", seed)
    enc, heads6 = enc.eval(), heads6.eval()
    nets = {h: MP.load_head(h, seed) for h in heads if h != "p6zero"}
    rows = {h: MP.Rows() for h in heads}
    lall = {h: Lall() for h in heads}
    with torch.no_grad():
        for b in dl:
            h, sv = MP.encode(enc, b, "cpu")
            z6 = heads6.assoc(h, b["pa"].float(), sv) if "p6zero" in rows else None
            outs = {n: (net(b, h, sv, b["pa"]) if net.arm != "z" else net(b)) for n, net in nets.items()}
            N = h.shape[1]
            for r in range(len(b["frame"])):
                k = int(b["frame"][r])
                if "p6zero" in rows:
                    ap = MP.p6_assign(z6, r, N)
                    lanes, route = MP.read_through(b, r, ap)
                    rows["p6zero"].add(k, b, r, ap, lanes, route)
                    lall["p6zero"].add(k, b, r, lanes)
                for n, o in outs.items():
                    ap = None
                    if "assoc" in o:
                        ap = o["assoc"][r].argmax(-1).numpy()
                        ap = np.where(ap == o["assoc"].shape[-1] - 1, -1, ap)
                    rows[n].add(k, b, r, ap, o["lanes"][r].argmax(-1).numpy(), torch.sigmoid(o["route"][r]).numpy())
                    lall[n].add(k, b, r, o["lanes"][r].argmax(-1).numpy())
    keys = sorted(set().union(*[set(r_.l) for r_ in rows.values()]) | set().union(*[set(r_.a) for r_ in rows.values()]))
    return (rows, lall), keys, scene


def summarize(rl, keys):
    rows, lall = rl
    out = {}
    for h, r in rows.items():
        d = {m: r.metric(m, keys) for m in MP.METRICS}
        d.update(extra_metrics(r, keys))
        d["lanes_all"] = lall[h].acc(keys)
        d["tokens"] = int(sum(len(v[1]) for v in r.a.values())) if r.a else 0
        d["lanes_tokens"] = int(sum(len(v[1]) for v in r.l.values()))
        out[h] = {k: (None if isinstance(v, float) and v != v else v) for k, v in d.items()}
    return out


def spec_list(modes):
    """-> [(name, spec)]: the exact map, then every mode of the catalogue at each level of its grid on an otherwise exact map, then the two deciding mixtures"""
    out = [("exact", {})]
    for c in ME.CATALOGUE:
        if modes != ["all"] and c["mode"] not in modes:
            continue
        for v in c["grid"]:
            out.append((f"{c['mode']}={v}", {c["key"]: v}))
    if modes == ["all"] or "dec" in modes:
        out.append(("dec", dict(ME.DECIDING["right"])))
    if modes == ["all"] or "wrong" in modes:
        out.append(("wrong_scene+dec", dict(ME.DECIDING["right"], wrong_scene=1)))
    return out


def degrade(seed, split, modes, every, heads, tag="", log=print):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"degrade_{split}_s{seed}{'_' + tag if tag else ''}.json")
    assert not os.path.exists(path), f"never overwritten: {path}"
    plan = plan_for(split, seed)
    res = dict(seed=seed, split=split, every=every, heads=heads, levels={})
    for name, spec in spec_list(modes):
        t0 = time.time()
        rows, keys, scene = level_rows(split, seed, spec, every, heads, plan, log=log)
        res["levels"][name] = dict(spec=spec, frames=len(keys), heads=summarize(rows, keys))
        L = res["levels"][name]["heads"]
        log(f"{name}: " + " | ".join(f"{h} assoc {v['assoc_acc'] if v['assoc_acc'] is not None else float('nan'):.3f} wrong_assign {v['wrong_assign'] if v['wrong_assign'] is not None else float('nan'):.3f} lanes_all {v['lanes_all'] if v['lanes_all'] is not None else float('nan'):.3f}" for h, v in L.items())
            + f" ({time.time() - t0:.0f} s)", flush=True)
        json.dump(res, open(path + ".part", "w"), indent=1)
    os.replace(path + ".part", path)
    log(f"written {os.path.relpath(path, ROOT)}")
    return res


def selftest():
    r = MP.Rows()
    r.a = {0: (np.array([1, -1, 2, 3]), np.array([1, -1, -1, 4]), np.array([0, 1, 2, 0]))}
    e = extra_metrics(r, [0])
    assert abs(e["wrong_assign"] - 0.5) < 1e-9 and abs(e["assign_rate"] - 0.75) < 1e-9, e                      # token 3 (a false one given segment 2) and token 4 (given 3 instead of 4) are wrong overwrites
    sp = dict(spec_list(["all"]))
    assert "exact" in sp and sp["shift=4.0"] == {"shift": 4.0} and sp["wrong_scene=1"] == {"wrong_scene": 1} and sp["dec"]["pose"] == 1.0 and sp["wrong_scene+dec"]["wrong_scene"] == 1
    assert [n for n, _ in spec_list(["yaw"])] == ["exact"] + [f"yaw={v}" for v in ME.CATALOGUE[2]["grid"]]
    print("maperror_probe selftest ok")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("degrade", "selftest"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="holdout", choices=("holdout", "val"))
    ap.add_argument("--modes", default="all")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--heads", default="p6zero,F1,z")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    return degrade(a.seed, a.split, a.modes.split(","), a.every, a.heads.split(","), a.tag)


if __name__ == "__main__":
    main()
