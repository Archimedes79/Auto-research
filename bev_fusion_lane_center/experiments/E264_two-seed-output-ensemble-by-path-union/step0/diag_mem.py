"""read-only CPU step-0 proxy for the E264 card: does a streaming log-odds memory (fuse_prior constants, all valid nodes of t-1 with fused pi >= 0.3, recursive)
separate true from false FAR raw nodes better than the single-frame validity?  node truth = within 1 m of a GT polyline (no assembly).
prints AUC (far nodes) and the true-node count at the false-node count of tau 0.7; null = memory from a frame of another scene."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import evaluate as EV
R_ = os.environ["BFLC_ROOT"]
reg = EV.registry()
rows = np.array(json.load(open(f"{R_}/data/splits.json"))["val"])
P = dict(np.load(os.path.expanduser("~/exp_gcar_b123/poses_val.npz")))
row2k = {int(r): k for k, r in enumerate(rows)}
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
sg = lambda x: 1 / (1 + np.exp(-x))

def dense(pl, step=0.25):
    out = []
    for p in pl:
        p = np.asarray(p, float)
        if len(p) == 1:
            out.append(p); continue
        for a, b in zip(p[:-1], p[1:]):
            n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
            out.append(a + (b - a) * np.linspace(0, 1, n + 1)[:, None])
    return np.concatenate(out) if out else np.zeros((0, 2))

def warp(pos, th, rp, rc):
    pw = pos @ P["R"][rp].T + P["t"][rp]
    pc = (pw - P["t"][rc]) @ P["R"][rc]
    u = np.stack([np.cos(th), np.sin(th)], 1) @ P["R"][rp].T @ P["R"][rc]
    return pc, np.arctan2(u[:, 1], u[:, 0])

def fuse(r, prior):
    pi = r["pi"].astype(np.float64).copy()
    if prior is None or len(prior[0]) == 0:
        return pi
    pp, pth, ppi = prior
    d = np.sqrt(((r["pos"][:, None, :] - pp[None]) ** 2).sum(-1))
    dth = np.abs(np.angle(np.exp(1j * (r["th_travel"][:, None] - pth[None]))))
    ok = (d < 1.5) & (dth < np.radians(30.0))
    d = np.where(ok, d, 1e9)
    im = d.argmin(1)
    m = d[np.arange(len(pi)), im] < 1e8
    f = sg(0.9 * lg(0.9 * ppi[im]) + 1.0 * lg(pi))
    pi = np.where(m, f, pi)
    par = r["parent"]
    order = np.argsort(r["level"], kind="stable")  # parents before children
    for i in order:  # the path validity stays a running minimum along the tree
        j = int(par[i])
        if 0 <= j < len(pi) and j != i:
            pi[i] = min(pi[i], pi[j])
    return pi

def auc(s, y):
    o = np.argsort(s); rk = np.empty(len(s)); rk[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return (rk[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

name = sys.argv[1]
raw, gts, _ = EV.builder_raw(reg[name], "val", require_cache=True)
order = sorted(range(len(rows)), key=lambda k: (int(P["scene"][rows[k]]), float(P["ts"][rows[k]])))
rng = np.random.default_rng(0)
out = {}
for mode in ("single", "memory", "null"):
    state = {}
    S, Y = [], []
    for k in order:
        r = raw[k]; row = int(rows[k])
        prior = None
        if mode == "memory":
            pr = int(P["prev"][row])
            if pr in row2k and row2k[pr] in state:
                pp, pth, ppi = state[row2k[pr]]
                pc, thc = warp(pp, pth, pr, row)
                prior = (pc, thc, ppi)
        elif mode == "null":  # a frame of another scene, its own (non-warped) memory state
            cand = [kk for kk in state if int(P["scene"][rows[kk]]) != int(P["scene"][row])]
            if cand:
                kk = cand[rng.integers(len(cand))]
                prior = state[kk]
        pi = fuse(r, prior) if mode != "single" else r["pi"].astype(np.float64)
        v = r["valid"] > 0.5
        keep = v & (pi >= 0.3)
        state[k] = (r["pos"][keep].astype(np.float64), r["th_travel"][keep].astype(np.float64), pi[keep])
        far = v & (np.linalg.norm(r["pos"], axis=1) >= 40)
        if far.any():
            gp = dense(gts[k]["pts"])
            dd = cKDTree(gp).query(r["pos"][far])[0] if len(gp) else np.full(far.sum(), 1e9)
            S.append(pi[far]); Y.append(dd <= 1.0)
    s, y = np.concatenate(S), np.concatenate(Y)
    out[mode] = (s, y)
    print(f"{name} {mode}: far valid nodes {len(s)}, true share {y.mean():.3f}, AUC {auc(s, y):.4f}, admitted at 0.7: true {int((y & (s >= 0.7)).sum())} false {int((~y & (s >= 0.7)).sum())}")
s0, y0 = out["single"]
F0 = int((~y0 & (s0 >= 0.7)).sum())
for mode in ("memory", "null"):
    s, y = out[mode]
    thr = np.sort(s[~y])[::-1][F0 - 1] if F0 > 0 else 1.0  # the threshold that admits the same number of false far nodes
    print(f"  {mode} at the false count of single tau 0.7 ({F0}): thr {thr:.3f}, true far nodes {int((y & (s >= thr)).sum())} vs single {int((y0 & (s0 >= 0.7)).sum())} (of {int(y0.sum())} true valid)")
