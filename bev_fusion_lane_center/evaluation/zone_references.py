"""E282: the REFERENCE ROWS of the zone metrics (card E282 section 1 and Layer 1): what do trivial rows, the stored baseline B and the stored E276 arm s score on `zone_metrics`? CPU, no GPU.

    wsl env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 bash tools/env/_py14.sh zone_references.py [--frames 6] [--s-frames 60] [--no-s] [--refresh]

VAL frames `val_frames()[::--frames]` (default every 6th frame, as the E275 dumps `temp/e275/pred_e273_p_s<seed>.pkl` were read), lane GT of `frame.lane()`, zones of the sparse target (0.3 m / 7.5 m).
Rows: (1) the sparse GT nodes, (2) every second dense node with a random phase per piece (seed 0), (3) a greedy 3 m step on the dense GT (anchors kept first, then the dense nodes in point-index order
unless a kept node lies closer than 3 m), (4) B plain at e >= 0.5 and (5) B + the hard close-node step 2.0 m (`qgraph_readout.suppress_close_nodes`), seeds 0 and 1 (E275 dumps), (6) the E276 arm s
checkpoint `runs/e276_s_s0_p3` (trained WITH the tier switches: a reference only) on --s-frames VAL frames spread over the split (CPU forward of the node head, cached in temp/e282/coder/s_nodes.npz):
at e >= 0.5, at the threshold that delivers 14.4 +- 2 nodes per 100 m on those frames (in-sample), and at e >= 0.3 plus the greedy 3 m step on the delivered nodes (by e descending).
LAYER 1: on the first 60 VAL frames (the frames of the E276 diagnostic `temp/e276/diag_s.sh`) the density at e >= 0.5 / 0.3 / 0.2 must reproduce the stored cards `runs/e276diag/*` within 3 %.
Prints the table (markdown) and writes it as json to temp/e282/references.json. All numbers are VAL, in-sample, one lane GT store (its signature is in the json).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

import bflc_paths as BP  # project paths from the location of this checkout
import qgraph_eval as QE
import qgraph_readout as RO
import qzones as QZ
import zone_metrics as ZM

SPARSE = (0.3, 7.5)
S_RUN = "e276_s_s0_p3"
DIAG = (0.5, 0.3, 0.2)                                                             # the thresholds of the E276 diagnostic (3.6 / 50.0 / 71.8 nodes per 100 m)
DIAG_FRAMES = 60                                                                   # `diag_s.sh --n 60`: the first 60 VAL frames
TOL_L1 = 0.03
SHOW = ("nodes_per_100m", "free_zone_recall", "free_zone_exact", "fixed_zone_recall", "fixed_zone_exact", "zone_recall", "zone_exact", "off_zone")


# ------------------------------------------------------------------------------------------------------------------------------ frames
class Prep:
    """per VAL frame the zones of its lane GT and the GT lane length in the metrics' box (memoised; the lane GT itself is let go of again)"""

    def __init__(self):
        self.done = {}

    def get(self, f):
        if f.k not in self.done:
            g = f.lane()
            self.done[f.k] = dict(zs=QZ.zones(g, SPARSE), len=QE.LaneRef(g).length)
            f._lane = f._ref = None
        return self.done[f.k]


def score_row(name, preps, Ps):
    """-> dict(name, frames, zones, the pooled numbers, the summed counts) of the delivered nodes Ps[i] (n, 2) of every frame"""
    cs = [ZM.frame_counts(p["zs"], p["len"], P) for p, P in zip(preps, Ps)]
    return dict(name=name, frames=len(cs), **ZM.pool(cs), counts={k: int(sum(c[k] for c in cs)) for k in ZM.COUNTS})


# ------------------------------------------------------------------------------------------------------------------------------ rows made from the GT
def sparse_gt(zs):
    return zs["nd"]["pos"][zs["gt_node"]]


def every_second(zs, rng):
    nd = zs["nd"]
    out = [nd["pos"][np.flatnonzero(nd["piece"] == p)[rng.randint(2)::2]] for p in np.unique(nd["piece"])]       # a random phase per piece
    return np.concatenate(out) if out else np.zeros((0, 2))


def greedy_step(pos, first, radius=3.0):
    """the nodes `first` are kept, then the others in index order unless a kept node lies closer than `radius` (exactly `radius` stays, as in qgraph_readout)"""
    keep = [int(m) for m in first]
    done = set(keep)
    for m in range(len(pos)):
        if m not in done and (not keep or np.min(np.linalg.norm(pos[keep] - pos[m], axis=1)) >= radius):
            keep.append(m)
    return pos[keep]


def greedy_dense(zs):
    return greedy_step(zs["nd"]["pos"], zs["gt_node"][zs["kind"] == 0])               # the anchors (fixed zones) first


# ------------------------------------------------------------------------------------------------------------------------------ rows made from stored predictions
def dump_rows(seed, frames):
    """(B plain e >= 0.5, B + hard step 2.0 m) delivered nodes per frame from the E275 dump of the seed"""
    dump = pickle.load(open(os.path.join(BP.ROOT, "temp", "e275", f"pred_e273_p_s{seed}.pkl"), "rb"))
    plain, hard = [], []
    for f in frames:
        d = dump[f.k]
        assert d["row"] == f.row, (f.k, d["row"], f.row)
        sel = d["e"] >= 0.5
        plain.append(d["pos"][sel])
        hard.append(d["pos"][RO.suppress_close_nodes(d["pos"], d["e"], 2.0, 0.5)])
    return plain, hard


def s_nodes(rows, cache_path, refresh):
    """-> {row: (e (2500,), pos (2500, 2))}: the existence probability and position of every cell of the node head of the arm s checkpoint (CPU, batch 1, no_grad); cached"""
    cache = {}
    if os.path.exists(cache_path) and not refresh:
        z = np.load(cache_path)
        cache = {int(r): (e, p) for r, e, p in zip(z["rows"], z["e"], z["pos"])}
    todo = [r for r in rows if r not in cache]
    if todo:
        import torch
        import lane_gt_data as LD
        import qpredict as QP
        torch.set_num_threads(2)
        ds = LD.QData("val_clean", pixels=False, map_mode="exact")
        row2i = {int(ds.bd.rows[k]): i for i, k in enumerate(ds.ks)}
        model = QP.load_model(os.path.join(BP.RUNS, S_RUN), "cpu")
        t0 = time.time()
        for n, r in enumerate(todo):
            x = torch.from_numpy(np.asarray(ds[row2i[r]]["x"], np.float32))[None]
            with torch.no_grad():
                no = model.node_stage(model.pyr(x))
            cache[r] = (no["e"][0].sigmoid().numpy().astype(float).reshape(-1), no["pos"][0].permute(1, 2, 0).numpy().astype(float).reshape(-1, 2))
            if n % 10 == 9 or n == len(todo) - 1:
                print(f"  arm s forward: {n + 1}/{len(todo)} frames, {(time.time() - t0) / (n + 1):.1f} s per frame", flush=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ks = sorted(cache)
        np.savez(cache_path, rows=np.array(ks), e=np.stack([cache[k][0] for k in ks]), pos=np.stack([cache[k][1] for k in ks]))
    return cache


def table(rows, header):
    out = [header, "| row | frames | " + " | ".join(("nodes/100 m", "free rec", "free exact", "fixed rec", "fixed exact", "all rec", "all exact", "off-zone")) + " |", "|---|---:|" + "---:|" * len(SHOW)]
    for r in rows:
        out.append(f"| {r['name']} | {r['frames']} | " + " | ".join(f"{r[k]:.1f}" if k == "nodes_per_100m" else f"{r[k]:.3f}" for k in SHOW) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, default=6, help="every N-th VAL frame for the rows made from the GT and the E275 dumps")
    ap.add_argument("--s-frames", type=int, default=60, help="frames spread over the split for row (6)")
    ap.add_argument("--no-s", action="store_true", help="skip row (6) and Layer 1 (no forward pass)")
    ap.add_argument("--refresh", action="store_true", help="recompute the cached forward pass of the arm s checkpoint")
    ap.add_argument("--out", default="temp/e282/references.json", help="json file, relative to the problem folder")
    a = ap.parse_args()
    a.out = os.path.join(BP.ROOT, a.out)                                              # the launcher runs a script from its own folder: no path relative to the cwd

    vals = QE.val_frames()
    assert QE.stored_share(vals) == 1.0, "not every VAL frame has a stored lane GT"
    prep = Prep()
    frames = vals[::a.frames]
    preps = [prep.get(f) for f in frames]
    nz = (sum(len(p["zs"]["kind"]) - int(p["zs"]["kind"].sum()) for p in preps), sum(int(p["zs"]["kind"].sum()) for p in preps))
    print(f"{len(frames)} VAL frames (every {a.frames}th), {nz[0]} fixed and {nz[1]} free zones, GT lane {sum(p['len'] for p in preps):.0f} m in the box", flush=True)
    rng = np.random.RandomState(0)
    rows = [score_row("sparse GT nodes", preps, [sparse_gt(p["zs"]) for p in preps]),
            score_row("every 2nd dense node (random phase, seed 0)", preps, [every_second(p["zs"], rng) for p in preps]),
            score_row("greedy 3 m step on the dense GT (anchors first)", preps, [greedy_dense(p["zs"]) for p in preps])]
    for seed in (0, 1):
        plain, hard = dump_rows(seed, frames)
        rows.append(score_row(f"B plain e>=0.5 (seed {seed})", preps, plain))
        rows.append(score_row(f"B + hard step 2.0 m (seed {seed})", preps, hard))
    res = dict(meta=dict(split="VAL", frames_stride=a.frames, n_frames=len(frames), sparse=list(SPARSE), zone_tol=ZM.ZONE_TOL, fixed_zones=nz[0], free_zones=nz[1], s_run=S_RUN,
                         in_sample=True), rows=rows, layer1=None, s_rows=[])
    try:
        sys.path.insert(0, BP.REPO)
        from GtGenerator import lane_gt
        res["meta"]["lane_signature"] = lane_gt.signature()
    except Exception as ex:  # noqa: BLE001 - the signature is documentation, not a reason to stop
        res["meta"]["lane_signature"] = f"unknown ({type(ex).__name__})"
    if not a.no_s:
        first = vals[:DIAG_FRAMES]
        spread = [vals[i] for i in np.linspace(0, len(vals) - 1, a.s_frames).astype(int)]
        cache = s_nodes(sorted({f.row for f in first + spread}), os.path.join(BP.ROOT, "temp", "e282", "coder", "s_nodes.npz"), a.refresh)
        # Layer 1: the density of the stored cards of the E276 diagnostic
        p1 = [prep.get(f) for f in first]
        l1 = []
        for thr in DIAG:
            r = score_row(f"arm s e>={thr:g}", p1, [cache[f.row][1][cache[f.row][0] >= thr] for f in first])
            stored = json.load(open(os.path.join(BP.RUNS, "e276diag", f"{S_RUN}.plain_t{thr:g}", "card.json")))["p"]["nodes"]["nodes_per_100m"]
            r.update(stored_nodes_per_100m=stored, rel_diff=abs(r["nodes_per_100m"] - stored) / stored)
            l1.append(r)
        res["layer1"] = dict(frames=len(first), tol=TOL_L1, ok=all(r["rel_diff"] <= TOL_L1 for r in l1), rows=l1)
        # row (6)
        p6 = [prep.get(f) for f in spread]
        E = [cache[f.row][0] for f in spread]
        P = [cache[f.row][1] for f in spread]
        mt = ZM.matched_threshold(E, [p["len"] for p in p6])
        s_rows = [score_row("arm s e>=0.5", p6, [p[e >= 0.5] for e, p in zip(E, P)]),
                  score_row(f"arm s e>={mt['thr']:.2f} (matched: {mt['dens']:.1f} per 100 m{'' if mt['ok'] else ', NOT within 2'})", p6, [p[e >= mt["thr"]] for e, p in zip(E, P)]),
                  score_row("arm s e>=0.3 + greedy 3 m step", p6, [p[RO.suppress_close_nodes(p, e, 3.0, 0.3)] for e, p in zip(E, P)])]
        rows += s_rows
        res["s_rows"], res["matched"] = s_rows, mt
    print(table(rows, f"\nReference rows, VAL, in-sample ({nz[1]} free and {nz[0]} fixed zones in the {len(frames)} frames of rows 1-5{'' if a.no_s else f'; the arm s rows: {a.s_frames} frames spread over the split'})"))
    if res["layer1"]:
        l1 = res["layer1"]
        print(table(l1["rows"], f"\nLayer 1: arm s on the first {l1['frames']} VAL frames (the E276 diagnostic frames)"))
        for r in l1["rows"]:
            print(f"  {r['name']}: {r['nodes_per_100m']:.3f} nodes per 100 m, stored card {r['stored_nodes_per_100m']:.3f}, relative difference {100 * r['rel_diff']:.2f} %")
        print("LAYER 1:", "OK (within 3 %)" if l1["ok"] else "FAILED (more than 3 % off the stored cards)")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("written", a.out)
    return 0 if (res["layer1"] is None or res["layer1"]["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
