"""Vertex-level evaluation of the grid chain (E271): the vertices of a system against the ground-truth lanes, at an equal vertex budget.

    wsl bash tools/env/_py14.sh vertex_eval.py ref [--name recall_sp0_dp] [--split val] [--n 0]     # r: the node ids of the delivered graph of a registry system
    wsl bash tools/env/_py14.sh vertex_eval.py null [--per100 11] [--seed 0]                          # uniform random vertices at the same budget (the null)
    wsl bash tools/env/_py14.sh vertex_eval.py head --run e271_a_s0 [--budget 11.2 8 14.3]                     # re-score a trained head (the budget of the card = r's vgt100)
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


def inside(V, *arrays):
    """vertices inside the scored box |x|, |y| < BOX (the ground truth lies there only: a vertex in the 2 m border of the 100 m raster could never be right), with the arrays that go with them"""
    V = np.asarray(V, np.float64).reshape(-1, 2)
    k = (np.abs(V[:, 0]) < BOX) & (np.abs(V[:, 1]) < BOX)
    return (V[k],) + tuple(None if a is None else np.asarray(a)[k] for a in arrays)


def frame_counts(V, H, gt, ids=None):
    G, T = gt_dense(gt) if not isinstance(gt, tuple) else gt
    V, H, ids = inside(V, H, ids)
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
    with open(os.path.join(d, "card.json"), "w") as f:
        json.dump(card, f)
    return card, d


def val_frames(n=0):
    """(frame positions ks, ground-truth graphs, scene id per frame) of the VAL split through BuilderData.rows"""
    import p16_data as PD
    ds = PD.BuilderData("val_clean", train=False)
    ks = list(range(len(ds.rows)))[: n or None]
    pz = np.load(f"{PD.D}/poses_val.npz")
    return ks, [ds.gt[int(ds.rows[k])] for k in ks], [int(pz["scene"][int(ds.rows[k])]) for k in ks]


def poly_vertices(g, step=5.0):
    """informational row r_poly: vertices every `step` m along the delivered polylines (what the graph covers between its nodes), with the polyline tangent"""
    V, H = [], []
    for p in g["pts"]:
        q = np.asarray(p, np.float64)
        if len(q) < 2:
            continue
        seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
        s_ = np.concatenate([[0.0], np.cumsum(seg)])
        if s_[-1] < 1e-6:
            continue
        t = np.arange(0.0, s_[-1] + 1e-9, step)
        pts = np.stack([np.interp(t, s_, q[:, 0]), np.interp(t, s_, q[:, 1])], 1)
        tan = np.gradient(pts, axis=0) if len(pts) > 1 else np.zeros((1, 2))
        tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        V.append(pts), H.append(tan)
    return (np.concatenate(V), np.concatenate(H)) if V else (np.zeros((0, 2)), np.zeros((0, 2)))


def run_ref(name, split, n):
    import evaluate as EV
    import lane_prototype as LP
    import sparse_metrics as SM
    ks, gts, scenes = val_frames(n)
    e = EV.registry()[name]
    g, conf, gts2, info = LP.delivered(e, split, ks)
    cs, cp = [], []
    for gi, gt in zip(g, gts2):
        V, H, ids = node_vertices(gi)
        cs.append(frame_counts(V, H, gt, ids))
        Vp, Hp = poly_vertices(gi)
        cp.append(frame_counts(Vp, Hp, gt))
    write_card("r_poly", cp, scenes, extra=dict(registry_name=name, note="informational: vertices every 5 m along the delivered polylines"))
    c10 = []
    for gi, gt in zip(g, gts2):
        Vp, Hp = poly_vertices(gi, 10.0)
        c10.append(frame_counts(Vp, Hp, gt))
    front = {}
    for st in (6.0, 8.0, 10.0, 12.0, 15.0, 20.0):                                       # the frontier of the simple baseline: coverage against vertices per 100 m
        cf = [frame_counts(*poly_vertices(gi, st), gt) for gi, gt in zip(g, gts2)]
        front[str(st)] = {k: float(v) for k, v in M.vertex_numbers(total(cf)).items()}
    write_card("r_poly10", c10, scenes, extra=dict(registry_name=name, frontier_by_step_m=front, note="the strong simple baseline: vertices every 10 m along the delivered polylines (frontier: every 6 .. 20 m)"))
    card, d = write_card("r", cs, scenes, extra=dict(registry_name=name, vert100_sparse_lanes=SM.vert100(g), asm_ms=info.get("asm_ms_cpu")))
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
    assert split in ("train", "val_clean"), f"{split!r}: E271 reads TRAIN and VAL only (hard rule 1: TEST is never touched here)"
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


def peak_weights(sparse, floor=0.01):
    """weights of the 3 x 3 maxima of a sparse map (5, X, Y) inside the scored box: the candidates of the read-out (the set does not depend on the threshold)"""
    import torch
    import torch.nn.functional as F
    w = torch.as_tensor(sparse)[0][None, None]
    peak = (w == F.max_pool2d(w, 3, 1, 1))[0, 0] & (w[0, 0] >= floor)
    n = peak.shape[0]
    c = (torch.arange(n, dtype=torch.float32) + 0.5) - n / 2
    inb = ((c.abs() < BOX)[:, None]) & ((c.abs() < BOX)[None, :])               # cell centres inside the box (1 m cells, origin -n/2)
    return w[0, 0][peak & inb].numpy()


def fit_threshold(model, budgets, n=300, seed=0, device="cuda"):
    """read-out thresholds of THIS model on TRAIN frames (in-sample, as the card says), one per budget (vertices per 100 m of GT lane): the k-th largest peak weight of the
    `n` frames, k = budget x (GT length of the frames) / 100. Exact, one pass; the vertices per 100 m actually reached on those frames are returned (checked within 0.5)"""
    import torch
    GD, GH, SL = _grid_imports()
    ds0 = GD.GridData("train")
    rng = np.random.RandomState(seed)
    ks = sorted(rng.choice(len(ds0), n, replace=False).tolist())
    ds, dl = GD.loader("train", bs=8, workers=2, shuffle=False, ks=ks)
    model.eval().to(device)
    ws, glen = [], 0.0
    with torch.no_grad():
        for b in dl:
            sp = SL.sparsify(model(b["x"].to(device).float())).cpu()
            for i in range(sp.shape[0]):
                ws.append(peak_weights(sp[i]))
            glen += float(b["m05"].sum()) * 0.5
    ws = np.sort(np.concatenate(ws))[::-1]
    out = {}
    for bud in budgets:
        k = int(round(bud * glen / 100.0))
        k = min(max(k, 1), len(ws))
        thr = float(ws[k - 1])
        reached = 100.0 * float((ws >= thr).sum()) / glen
        out[bud] = (thr, reached)                                              # a row is only a row of that budget when it was REACHED (the weights of the head may not allow it)
    return out


def eval_head(model, name, seed, budgets=(12.8,), device="cuda", n=0, extra=None):
    """score a trained head on VAL at the budgets: the card of the arm at the first budget (runs/e271/<name>/card.json), the frontier (all budgets) inside it"""
    GD, GH, SL = _grid_imports()
    import p16_data as PD
    fit = fit_threshold(model, list(budgets), device=device)
    thr_of = {b: fit[b][0] for b in budgets}
    nval = len(PD.BuilderData("val_clean", train=False).rows)
    ks = list(range(nval))[: n or None]
    res, gts, rows = infer_vertices(model, "val_clean", ks, sorted(set(thr_of.values())), device)
    pz = np.load(f"{PD.D}/poses_val.npz")
    scenes = [int(pz["scene"][r]) for r in rows]
    cards = {}
    for b in budgets:
        cs = [frame_counts(V, H, g) for (V, H), g in zip(res[thr_of[b]], gts)]
        cards[b] = (M.vertex_numbers(total(cs)), cs)
    b0 = budgets[0]
    front = {str(b): dict(thr=thr_of[b], train_vgt100=fit[b][1], reached=bool(abs(fit[b][1] - b) <= 0.5), **{k: float(v) for k, v in cards[b][0].items()}) for b in budgets}
    card, d = write_card("a", cards[b0][1], scenes, seed=seed, extra=dict(budget=b0, thr=thr_of[b0], frontier=front, **(extra or {})), name=name)
    return card, d


def paired_bootstrap(card_a, card_c, reps=2000, seed=0, lo=2.5, hi=97.5):
    """paired scene bootstrap of a - c for cov5, prec1, vgt100 (the frames of a scene are resampled together; the numbers are ratios of summed counts), from the per-frame counts
    stored in the two cards (same frames, same order). -> {metric: {"lo", "hi", "mean"}}: the format of `compare.paired` that decide.py reads"""
    assert card_a["scenes"] == card_c["scenes"], "the two cards are not on the same frames"
    sc = np.asarray(card_a["scenes"])
    ids = np.unique(sc)
    A = {k: np.asarray([c[k] for c in card_a["counts"]], np.float64) for k in ("cov", "prec")}
    C = {k: np.asarray([c[k] for c in card_c["counts"]], np.float64) for k in ("cov", "prec")}
    per = {i: np.nonzero(sc == i)[0] for i in ids}

    def nums(X, ix):
        cov, ok = X["cov"][ix].sum(0), X["prec"][ix].sum(0)
        return np.array([cov[0] / cov[1], ok[0] / ok[1] if ok[1] > 0 else 1.0, 100.0 * ok[1] / cov[1]])
    rng = np.random.RandomState(seed)
    d = np.zeros((reps, 3))
    for r in range(reps):
        ix = np.concatenate([per[i] for i in rng.choice(ids, len(ids))])
        d[r] = nums(A, ix) - nums(C, ix)
    full = nums(A, np.arange(len(sc))) - nums(C, np.arange(len(sc)))
    return {m: dict(lo=float(np.percentile(d[:, j], lo)), hi=float(np.percentile(d[:, j], hi)), mean=float(full[j])) for j, m in enumerate(("cov5", "prec1", "vgt100"))}


def write_compare(run, control="r"):
    """add `compare` (name of the control card, paired scene bootstrap) to the card of a run, so that decide.py reads the interval"""
    pa = os.path.join(ROOT, "runs", "e271", run, "card.json")
    pc = os.path.join(ROOT, "runs", "e271", control, "card.json")
    a, c = json.load(open(pa)), json.load(open(pc))
    a["compare"] = dict(name=c["name"], paired=paired_bootstrap(a, c), reps=2000, how="paired scene bootstrap, 95 %, a - control")
    with open(pa, "w") as f:
        json.dump(a, f)
    return a["compare"]


def verdict(runs=("e271_a_s0", "e271_a_s1"), control="r", band=0.03, prec_loss=0.02, vgt_loss=1.0):
    """the written rule of the E271 card applied BY HAND (decide.py refuses vertex-level cards: card.validate demands the project's graph headline metrics in every card):
    headline cov5 a - r on the mean of the seeds, band 0.03 (a tie inside it); guards prec1 (loss <= 0.02) and vgt100 (gain <= 1.0 per 100 m); every seed on the same side.
    -> dict(verdict, text, ...) with KEEP (> +band), TIE (inside), CUT (< -band or a guard violated); INCONCLUSIVE when the seeds disagree in sign"""
    c = json.load(open(os.path.join(ROOT, "runs", "e271", control, "card.json")))["metrics"]
    cards = [json.load(open(os.path.join(ROOT, "runs", "e271", r, "card.json"))) for r in runs]
    eff = {m: [cd["metrics"][m] - c[m] for cd in cards] for m in ("cov5", "prec1", "vgt100")}
    mean = {m: float(np.mean(v)) for m, v in eff.items()}
    guards = []
    if mean["prec1"] < -prec_loss:
        guards.append(f"prec1 {mean['prec1']:+.3f} < -{prec_loss}")
    if mean["vgt100"] > vgt_loss:
        guards.append(f"vgt100 {mean['vgt100']:+.2f} > +{vgt_loss}")
    signs = {np.sign(x) for x in eff["cov5"] if abs(x) > band} or {0}
    if guards:
        v = "CUT"
    elif mean["cov5"] > band and all(x > band for x in eff["cov5"]):
        v = "KEEP"
    elif mean["cov5"] < -band:
        v = "CUT"
    elif len(signs) > 1:
        v = "INCONCLUSIVE"
    elif abs(mean["cov5"]) <= band:
        v = "TIE"
    else:
        v = "INCONCLUSIVE"
    text = (f"by hand: card.validate demands the project's graph headline metrics (Q, topo_F1, ...) in every card and a vertex-level card has none, so decide.py cannot read these cards; "
            f"the rule of the E271 card applied as written: cov5 a - r per seed {[round(x, 3) for x in eff['cov5']]} (mean {mean['cov5']:+.3f}, band {band}), prec1 {[round(x, 3) for x in eff['prec1']]}, "
            f"vgt100 {[round(x, 2) for x in eff['vgt100']]}; guards {'violated: ' + '; '.join(guards) if guards else 'held'}; verdict {v}")
    out = dict(verdict=v, text=text, effects=eff, mean=mean, guards=guards, runs=list(runs), control=control)
    with open(os.path.join(ROOT, "runs", "e271", "verdict.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def selftest():
    # the read-out finds the isolated peaks and nothing else, in metres
    s = np.zeros((5, 100, 100), np.float32)
    s[0, 10, 20], s[0, 10, 21] = 0.9, 0.8                      # two neighbours: one vertex (the maximum)
    s[0, 50, 50] = 0.2                                         # below the threshold
    s[1, 10, 20], s[2, 10, 20] = 1.0, 0.0
    s[3, 10, 20], s[4, 10, 20] = 0.25, -0.4
    V, H, w = read_out(s, 0.3)
    assert len(V) == 1 and np.allclose(V[0], [(10 + 0.5 + 0.25) - 50, (20 + 0.5 - 0.4) - 50]) and np.allclose(H[0], [1, 0]), (V, H)   # both offsets, the cell centre +0.5
    V2, _ = inside(np.array([[47.9, 0.0], [48.1, 0.0], [0.0, -48.5]]), None)
    assert len(V2) == 1, "the border band is dropped"
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
    # the paired bootstrap: identical cards -> a zero interval; a card with all vertices right against one with none -> a clear gain
    fr = lambda cov, ok, n: dict(cov=[cov, 10.0], prec=[ok, n])  # noqa: E731
    mk = lambda cs: dict(scenes=[0, 0, 1, 1, 2, 2], counts=cs)  # noqa: E731
    same = mk([fr(5, 3, 4)] * 6)
    z = paired_bootstrap(same, same, reps=200)
    assert all(abs(v["lo"]) < 1e-9 and abs(v["hi"]) < 1e-9 for v in z.values()), z
    better = mk([fr(9, 4, 4)] * 6)
    w = paired_bootstrap(better, same, reps=200)
    assert w["cov5"]["lo"] > 0.3 and w["prec1"]["lo"] > 0.0, w
    print("vertex_eval selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ref", "null", "head", "compare", "verdict", "selftest"])
    ap.add_argument("--control", default="r", help="compare: the control card (runs/e271/<name>/card.json)")
    ap.add_argument("--run", default=None, help="head: the run folder name under runs/ (model.pt, config.json)")
    ap.add_argument("--budget", type=float, nargs="*", default=[11.2, 8.0, 14.3], help="head: vertices per 100 m of GT lane, the first one is the card's")
    ap.add_argument("--name", default="recall_sp0_dp")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--per100", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    assert a.split == "val", "E271 is scored on VAL only"
    if a.cmd == "verdict":
        print(verdict()["text"])
        return
    if a.cmd == "compare":
        cmp = write_compare(a.run, a.control)
        print(a.run, "-", a.control, {m: {k: round(v, 4) for k, v in x.items()} for m, x in cmp["paired"].items()})
        return
    if a.cmd == "head":
        import torch
        GD, GH, SL = _grid_imports()
        run_dir = os.path.join(ROOT, "runs", a.run)
        model = GH.GridHead(width=json.load(open(os.path.join(run_dir, "config.json"))).get("width", 48))
        model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu")["state"])
        seed = int(json.load(open(os.path.join(run_dir, "config.json"))).get("seed", 0))
        card, d = eval_head(model, a.run, seed, budgets=tuple(a.budget), device="cuda" if torch.cuda.is_available() else "cpu", extra=dict(rescored=True))
        print(a.run, {k: round(v, 4) for k, v in card["metrics"].items()}, "thr", card["thr"], "->", os.path.relpath(d, ROOT))
        return
    card, d = run_ref(a.name, a.split, a.n) if a.cmd == "ref" else run_null(a.per100, a.seed, a.n)
    print(a.cmd, {k: round(v, 4) for k, v in card["metrics"].items()}, {k: card[k] for k in ("frames", "vert100_sparse_lanes") if k in card}, "->", os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
