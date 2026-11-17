import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_seq_data as LS
import lane_tok_data as LT
import lane_tokcache as LC
import p16_data as PD

seed = sys.argv[1] if len(sys.argv) > 1 else "0"
split = "val"
c = LC.TokCache(os.path.join(R, f"runs/e365_cache/be292_a_{seed}_K512_9ad1d4ce30_v1/{split}"))
pose = dict(np.load(f"{PD.D}/poses_{split}.npz"))
row2i = {int(r): i for i, r in enumerate(c.rows)}
ds = LT.TokData(split, res=2.0, feats=False)
RAD, COS = 1.5, 0.5
rng = np.random.default_rng(0)
acc = {}


def add(k, a, n=1):
    acc[k] = acc.get(k, 0) + a


def support(pos, dirv, ppos, pdir):
    if len(ppos) == 0:
        return np.zeros(len(pos), bool)
    d = np.linalg.norm(pos[:, None] - ppos[None], axis=2)
    return ((d <= RAD) & (dirv @ pdir.T >= COS)).any(1)


topo_cols = [0, 1, 2, 3, 4, 5]                      # stop, exit, split, merge, start, end
n = len(c)
for i in range(n):
    pr = int(pose["prev"][int(c.rows[i])])
    if pr < 0 or pr not in row2i:
        continue
    ip = row2i[pr]
    t = ds.table(i)
    gpos, gdir, gfl = t["pos"], t["dirv"], t["flags"]
    M = len(gpos)
    pos, dirv, valid, ep = (np.array(c.f[k][i], float) if k != "valid" else np.array(c.f[k][i]) for k in ("pos", "dirv", "valid", "e_p"))
    deliv = valid & (ep >= 0.5)
    weak = valid & ~deliv
    Rm, tv = LS.rel_pose(pose["R"][int(c.rows[i])], pose["t"][int(c.rows[i])], pose["R"][pr], pose["t"][pr])
    pdel = np.array(c.f["valid"][ip]) & (np.array(c.f["e_p"][ip], float) >= 0.5)
    pp = np.array(c.f["pos"][ip], float)[pdel]
    pd_ = np.array(c.f["dirv"][ip], float)[pdel]
    wpos, wdir = pp @ Rm.T.astype(float) + tv, pd_ @ Rm.T.astype(float)
    # null: a random other frame of the stream (identity pose)
    j = int(rng.integers(len(c)))
    rdel = np.array(c.f["valid"][j]) & (np.array(c.f["e_p"][j], float) >= 0.5)
    rpos, rdir = np.array(c.f["pos"][j], float)[rdel], np.array(c.f["dirv"][j], float)[rdel]
    if M == 0:
        continue
    near = lambda P, D_, V=None: (np.linalg.norm(P[:, None] - gpos[None], axis=2) <= RAD) & (D_ @ gdir.T >= COS)  # noqa: E731
    nd = near(pos, dirv) & deliv[:, None]
    found = nd.any(0)
    miss = ~found
    nw = near(pos, dirv) & weak[:, None]
    # GT node level
    past_sup = (near(wpos, wdir).any(0) if len(wpos) else np.zeros(M, bool))
    has_weak = nw.any(0)
    topo = gfl[:, topo_cols].any(1)
    add("gt", M)
    add("missed", int(miss.sum()))
    add("missed_weak_past", int((miss & has_weak & past_sup).sum()))
    add("missed_weak_past_topo", int((miss & has_weak & past_sup & topo).sum()))
    add("missed_weak_past_startend", int((miss & has_weak & past_sup & gfl[:, [4, 5]].any(1)).sum()))
    add("missed_topo", int((miss & topo).sum()))
    add("missed_startend", int((miss & gfl[:, [4, 5]].any(1)).sum()))
    add("gt_topo", int(topo.sum()))
    # weak tokens: novel (near a missed GT node), duplicate (near a found GT node only), false
    w_idx = np.flatnonzero(weak)
    if len(w_idx):
        nearw = near(pos[w_idx], dirv[w_idx])
        novel = (nearw & miss[None]).any(1)
        dup = nearw.any(1) & ~novel
        false = ~nearw.any(1)
        # a duplicate = near a found GT node; is it also near a delivered token (token level)?
        s_true = support(pos[w_idx], dirv[w_idx], wpos, wdir)
        s_id = support(pos[w_idx], dirv[w_idx], pp, pd_)
        s_rand = support(pos[w_idx], dirv[w_idx], rpos, rdir)
        for name, m in (("novel", novel), ("dup", dup), ("false", false)):
            add(f"{name}_n", int(m.sum()))
            add(f"{name}_pose", int((s_true & m).sum()))
            add(f"{name}_ident", int((s_id & m).sum()))
            add(f"{name}_rand", int((s_rand & m).sum()))
g = acc["gt"]
print("seed", seed, "VAL frames with a past: GT nodes", g, "missed", acc["missed"], "(%.4f)" % (acc["missed"] / g))
print("weak tokens by role: n | supported by the warped delivered past | identity pose | a random frame")
for name in ("novel", "dup", "false"):
    n_ = acc[f"{name}_n"]
    print("  %-6s %7d | %.3f | %.3f | %.3f" % (name, n_, acc[f"{name}_pose"] / n_, acc[f"{name}_ident"] / n_, acc[f"{name}_rand"] / n_))
tot_w = acc["novel_n"] + acc["dup_n"]
print("share of the weak 'true' tokens that are DUPLICATES of an already found GT node: %.3f" % (acc["dup_n"] / tot_w))
print("recall ceiling (missed, weak token near, past support) as node recall: +%.4f; of it on topology nodes (stop/exit/split/merge/start/end): +%.4f; start/end only: +%.4f" % (
    acc["missed_weak_past"] / g, acc["missed_weak_past_topo"] / g, acc["missed_weak_past_startend"] / g))
print("topology nodes: %.4f of all GT nodes; of the missed %.4f; start/end missed %.4f of all missed" % (acc["gt_topo"] / g, acc["missed_topo"] / acc["missed"], acc["missed_startend"] / acc["missed"]))
