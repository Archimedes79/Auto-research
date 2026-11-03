"""Vertex-level evaluation of the grid chain (E271): the vertices of a system against the ground-truth lanes, at an equal vertex budget.

    wsl bash tools/env/_py14.sh vertex_eval.py ref [--name recall_sp0_dp] [--split val] [--n 0]     # r: the node ids of the delivered graph of a registry system
    wsl bash tools/env/_py14.sh vertex_eval.py null [--per100 11] [--seed 0]                          # uniform random vertices at the same budget (the null)
    wsl bash tools/env/_py14.sh vertex_eval.py selftest

Every number is `lane_task.measures.vertex_counts` summed over the frames (cov5, prec1, vgt100: see there). A card is written to runs/e271/<arm>/card.json
(metrics, `better`, frames, the per-frame counts for a paired scene bootstrap). The rows of a split are taken through `BuilderData.rows` /
`data/splits.json`, never by iterating a stored file that also holds TEST rows (hard rule 1); only VAL here.
The READ-OUT of a trained head (`read_out`) is a named rule step on the output, not on the training path: cells with w >= thr that are also the 3 x 3 maximum.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")  # a shared machine
import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from lane_task import measures as M  # noqa: E402

BOX = 48.0                                                    # the ground-truth points of the batch tensors lie inside |x|, |y| < 48 m
BETTER = {"cov5": "higher", "prec1": "higher", "vgt100": "lower", "ms": "lower"}


def gt_dense(g, ds=0.5):
    """ground-truth graph (pts, edges) -> dense lane points (M, 2) every ds m inside the scored box, their travel tangents (M, 2)"""
    import graph_eval as GE
    P, T, _ = GE.dense_points(g, ds)
    P, T = np.asarray(P, np.float64).reshape(-1, 2), np.asarray(T, np.float64).reshape(-1, 2)
    k = (np.abs(P[:, 0]) < BOX) & (np.abs(P[:, 1]) < BOX)
    return P[k], T[k]


def node_vertices(g):
    """the vertices of a delivered graph: one per node id, with the tangent of the polyline at every occurrence (a node in several pieces has several rows, one id)"""
    V, H, ids = [], [], []
    for k, p in enumerate(g["pts"]):
        q = np.asarray(p, np.float64)
        if len(q) < 2:
            continue
        t = np.gradient(q, axis=0)
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nd = g["nodes"][k] if g.get("nodes") is not None and k < len(g["nodes"]) else [(k, j) for j in range(len(q))]
        for j in range(len(q)):
            V.append(q[j]), H.append(t[j]), ids.append(hash((k, j)) if not isinstance(nd[j], (int, np.integer)) else int(nd[j]))
    return np.asarray(V).reshape(-1, 2), np.asarray(H).reshape(-1, 2), np.asarray(ids, np.int64)


def read_out(sparse, thr, res=1.0, origin=(-50.0, -50.0)):
    """READ-OUT (rule step on the output): sparse map (5, X, Y) numpy / torch [w, cos, sin, ox, oy] -> vertices (N, 2) metres, headings (N, 2), weights (N,):
    cells with w >= thr that are the maximum of their 3 x 3 neighbourhood; axes as the feature raster (rows = x right, columns = y forward)"""
    import torch
    import torch.nn.functional as F
    s = torch.as_tensor(sparse)
    w = s[0][None, None]
    peak = (w == F.max_pool2d(w, 3, 1, 1))[0, 0] & (s[0] >= thr)
    ix, iy = torch.nonzero(peak, as_tuple=True)
    x = (ix.float() + 0.5 + s[3][ix, iy]) * res + origin[0]
    y = (iy.float() + 0.5 + s[4][ix, iy]) * res + origin[1]
    h = s[1:3][:, ix, iy].T
    h = h / h.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.stack([x, y], 1).numpy(), h.numpy(), s[0][ix, iy].numpy()


def frame_counts(V, H, gt, ids=None):
    G, T = gt_dense(gt) if not isinstance(gt, tuple) else gt
    return M.vertex_counts(V, H, G, T, ids=ids)


def total(cs):
    return {k: sum(c[k] for c in cs) for k in cs[0]}


def git_head():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_card(arm, cs, scenes=None, seed=0, extra=None, name=None):
    """runs/e271/<arm>/card.json: summed numbers + the per-frame counts (for a paired scene bootstrap)"""
    num = M.vertex_numbers(total(cs))
    card = dict(schema=1, name=name or arm, arm=arm, seed=seed, split="val", frames=len(cs), metrics={k: float(v) for k, v in num.items()},
                better={k: BETTER[k] for k in num}, commit=git_head(), date=time.strftime("%Y-%m-%d %H:%M"), counts=[{k: v.tolist() for k, v in c.items()} for c in cs])
    if scenes is not None:
        card["scenes"] = [int(s) for s in scenes]
    card.update(extra or {})
    d = os.path.join(ROOT, "runs", "e271", name or arm)
    os.makedirs(d, exist_ok=True)
    json.dump(card, open(os.path.join(d, "card.json"), "w"))
    return card, d


def val_frames(n=0):
    """(frame positions ks, ground-truth graphs, scene id per frame) of the VAL split through BuilderData.rows"""
    import p16_data as PD
    ds = PD.BuilderData("val_clean", train=False)
    ks = list(range(len(ds.rows)))[: n or None]
    import json as J
    pz = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
    return ks, [ds.gt[int(ds.rows[k])] for k in ks], [int(pz["scene"][int(ds.rows[k])]) for k in ks]


def run_ref(name, split, n):
    import evaluate as EV
    import lane_prototype as LP
    import sparse_metrics as SM
    ks, gts, scenes = val_frames(n)
    e = EV.registry()[name]
    g, conf, gts2, info = LP.delivered(e, split, ks)
    cs = []
    for gi, gt in zip(g, gts2):
        V, H, ids = node_vertices(gi)
        cs.append(frame_counts(V, H, gt, ids))
    card, d = write_card("r", cs, scenes, extra=dict(registry=name, vert100_sparse_lanes=SM.vert100(g), asm_ms=info.get("asm_ms_cpu")))
    return card, d


def run_null(per100, seed, n):
    rng = np.random.RandomState(seed)
    ks, gts, scenes = val_frames(n)
    cs = []
    for gt in gts:
        G, T = gt_dense(gt)
        k = int(round(per100 * len(G) * 0.5 / 100.0))
        V = rng.uniform(-BOX, BOX, size=(k, 2))
        a = rng.uniform(0, 2 * np.pi, k)
        cs.append(M.vertex_counts(V, np.stack([np.cos(a), np.sin(a)], 1), G, T))
    return write_card("null", cs, scenes, seed=seed, extra=dict(per100=per100))


# ------------------------------------------------------------------------------------------------------------------ a trained grid head
def _grid_imports():
    sys.path.insert(0, os.path.join(ROOT, "training", "grid"))
    sys.path.insert(0, os.path.join(ROOT, "training", "data"))
    import data as GD
    import head as GH
    import soft_lines as SL
    return GD, GH, SL


def infer_vertices(model, split, ks, thr_list, device="cuda", bs=8, workers=2):
    """run the head on the frames ks of the split; -> per threshold a list of (V, H) per frame, and the per-frame GT (G, T): one pass, several read-outs"""
    import torch
    GD, GH, SL = _grid_imports()
    ds, dl = GD.loader(split, bs=bs, workers=workers, shuffle=False, ks=ks)
    model.eval().to(device)
    res = {t: [] for t in thr_list}
    gts, rows = [], []
    with torch.no_grad():
        for b in dl:
            out = model(b["x"].to(device).float())
            sp = SL.sparsify(out).cpu()
            for i in range(sp.shape[0]):
                for t in thr_list:
                    V, H, _ = read_out(sp[i], t)
                    res[t].append((V, H))
                m = b["m05"][i] > 0
                gts.append((b["g05"][i][m].numpy().astype(np.float64), b["t05"][i][m].numpy().astype(np.float64)))
                rows.append(int(b["row"][i]))
    return res, gts, rows


def fit_threshold(model, budget, n=300, seed=0, device="cuda", lo=0.02, hi=0.98):
    """read-out threshold of THIS model on TRAIN frames (in-sample, as the card says) so that it delivers `budget` vertices per 100 m of GT lane (bisection, 12 steps)"""
    GD, GH, SL = _grid_imports()
    rng = np.random.RandomState(seed)
    ks = sorted(rng.choice(8414, n, replace=False).tolist())
    grid = [round(float(t), 4) for t in np.linspace(lo, hi, 25)]
    res, gts, _ = infer_vertices(model, "train", ks, grid, device)
    v = []
    for t in grid:
        cs = [frame_counts(V, H, g) for (V, H), g in zip(res[t], gts)]
        v.append(M.vertex_numbers(total(cs))["vgt100"])
    v = np.asarray(v)
    j = int(np.argmin(np.abs(v - budget)))
    return grid[j], dict(zip(map(str, grid), v.tolist()))


def eval_head(model, name, seed, budgets=(12.8,), device="cuda", n=0, extra=None):
    """score a trained head on VAL at the budgets: the card of the arm at the first budget (runs/e271/<name>/card.json), the frontier (all budgets) inside it"""
    cards = {}
    thr_of = {}
    for b in budgets:
        thr_of[b], _ = fit_threshold(model, b, device=device)
    ks = list(range(873))[: n or None]
    res, gts, rows = infer_vertices(model, "val_clean", ks, sorted(set(thr_of.values())), device)
    pz = np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz"))
    scenes = [int(pz["scene"][r]) for r in rows]
    for b in budgets:
        cs = [frame_counts(V, H, g) for (V, H), g in zip(res[thr_of[b]], gts)]
        cards[b] = (M.vertex_numbers(total(cs)), cs)
    b0 = budgets[0]
    front = {str(b): dict(thr=thr_of[b], **{k: float(v) for k, v in cards[b][0].items()}) for b in budgets}
    card, d = write_card("a", cards[b0][1], scenes, seed=seed, extra=dict(budget=b0, thr=thr_of[b0], frontier=front, **(extra or {})), name=name)
    return card, d


def selftest():
    # the read-out finds the isolated peaks and nothing else, in metres
    s = np.zeros((5, 100, 100), np.float32)
    s[0, 10, 20], s[0, 10, 21] = 0.9, 0.8                      # two neighbours: one vertex (the maximum)
    s[0, 50, 50] = 0.2                                         # below the threshold
    s[1, 10, 20], s[2, 10, 20] = 1.0, 0.0
    s[3, 10, 20] = 0.25
    V, H, w = read_out(s, 0.3)
    assert len(V) == 1 and np.allclose(V[0], [(10 + 0.5 + 0.25) - 50, (20 + 0.5) - 50]) and np.allclose(H[0], [1, 0]), (V, H)
    # a node in two pieces is one vertex; positions and headings of a graph
    g = dict(pts=[np.array([[0.0, 0.0], [0.0, 10.0]]), np.array([[0.0, 10.0], [5.0, 20.0]])], nodes=[[7, 8], [8, 9]])
    V, H, ids = node_vertices(g)
    assert len(V) == 4 and len(set(ids.tolist())) == 3
    # the null scores low, the ground truth as vertices scores high
    gt = dict(pts=[np.stack([np.zeros(80), np.linspace(-40, 40, 80)], 1)], edges=[])
    G, T = gt_dense(gt)
    c = M.vertex_numbers(M.vertex_counts(G[::10], T[::10], G, T))
    assert c["cov5"] > 0.99 and c["prec1"] == 1.0, c
    rng = np.random.RandomState(0)
    V = rng.uniform(-BOX, BOX, size=(len(G[::10]), 2))
    c = M.vertex_numbers(M.vertex_counts(V, None, G, T))
    assert c["cov5"] < 0.3 and c["prec1"] < 0.3, c
    print("vertex_eval selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ref", "null", "selftest"])
    ap.add_argument("--name", default="recall_sp0_dp")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--per100", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    assert a.split == "val", "E271 is scored on VAL only"
    card, d = run_ref(a.name, a.split, a.n) if a.cmd == "ref" else run_null(a.per100, a.seed, a.n)
    print(a.cmd, {k: round(v, 4) for k, v in card["metrics"].items()}, {k: card[k] for k in ("frames", "vert100_sparse_lanes") if k in card}, "->", os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
