import os
import pickle
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_tok_data as LT
import lane_tokcache as LC

run = sys.argv[1] if len(sys.argv) > 1 else "e373_cb_s0"
seed = int(run[-1])
thr_long, keep_thr = 0.2, 0.6
fr = pickle.load(open(os.path.join(R, "runs", run, "token_val.pkl"), "rb"))
frames = fr["frames"] if isinstance(fr, dict) else fr
c = LC.TokCache(os.path.join(R, f"runs/e365_cache/be292_a_{seed}_K512_9ad1d4ce30_v1/val"))
ds = LT.TokData("val", res=2.0, feats=False)
tok2i = {t: i for i, t in enumerate(c.tokens)}
RAD, COS = 1.5, 0.5
cnt = dict(total=0, tp=0, dup_of_true=0, wrong_pair=0, false_end=0, one_end_far=0)
pvals = {k: [] for k in cnt}
typ = {k: [] for k in cnt}
uu = {k: [] for k in cnt}
n_gt_long = 0
frames_n = 0
for f in frames:
    i = tok2i[f["token"]]
    t = ds.table(i)
    gpos, gdir = t["pos"], t["dirv"]
    pos, dirv = np.array(c.f["pos"][i], float), np.array(c.f["dirv"][i], float)
    gid = np.array(c.f["gt_id"][i])
    cd = f["cand"]
    src, dst, p, lg = (np.asarray(cd[k]) for k in ("src", "dst", "p", "long"))
    pnone = np.asarray(f["pnone"], float)
    sel = lg.astype(bool) & (p >= thr_long) & (pnone[src] < keep_thr) & (pnone[dst] < keep_thr)
    # GT long links as node-index pairs
    gl = set((int(a), int(b)) for a, b, l in zip(t["src"], t["dst"], t["long"]) if l)
    n_gt_long += len(gl)
    nearall = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
    frames_n += 1
    for s_, d_, p_ in zip(src[sel], dst[sel], p[sel]):
        cnt["total"] += 1
        a, b = int(gid[s_]), int(gid[d_])
        if a >= 0 and b >= 0 and (a, b) in gl:
            k = "tp"
        else:
            ns, nd = np.flatnonzero(nearall[s_]), np.flatnonzero(nearall[d_])
            if len(ns) == 0 or len(nd) == 0:
                k = "false_end"
            elif any((int(x), int(y)) in gl for x in ns for y in nd):
                k = "dup_of_true"
            else:
                k = "wrong_pair"
        cnt[k] += 1
        pvals[k].append(p_)
        uo, ui = np.array(c.f["uout"][i], float), np.array(c.f["uin"][i], float)
        uu[k].append((1 / (1 + np.exp(-uo[s_])), 1 / (1 + np.exp(-ui[d_]))))
        pt = np.asarray(f["ptype"], float)
        typ[k].append((pt[s_, 4], pt[d_, 5], pt[s_, 4] * pt[d_, 5]))
print(run, "VAL frames", frames_n, "| GT long links", n_gt_long, "| delivered long links (p >= 0.2, both ends kept p_none < 0.6):", cnt["total"])
for k in ("tp", "dup_of_true", "wrong_pair", "false_end"):
    v = np.array(pvals[k]) if pvals[k] else np.zeros(1)
    print("  %-12s %7d  share %.3f  median p %.2f" % (k, cnt[k], cnt[k] / max(cnt["total"], 1), np.median(v)))
print("  true links delivered (tp / GT long): %.3f" % (cnt["tp"] / n_gt_long))

print("typed factor p_entry(src) * p_exit(dst) of the head's own type softmax: median / share above 0.1 / above 0.3 per category")
for k in ("tp", "dup_of_true", "wrong_pair", "false_end"):
    a = np.array(typ[k]) if typ[k] else np.zeros((1, 3))
    print("  %-12s p_entry %.2f p_exit %.2f product median %.3f | > 0.1: %.3f | > 0.3: %.3f" % (k, np.median(a[:, 0]), np.median(a[:, 1]), np.median(a[:, 2]), (a[:, 2] > 0.1).mean(), (a[:, 2] > 0.3).mean()))

print("B' long-link evidence sigmoid(uout[src]) * sigmoid(uin[dst]): median per source / destination, share of the product above 0.1 / 0.3")
for k in ("tp", "dup_of_true", "wrong_pair", "false_end"):
    a = np.array(uu[k]) if uu[k] else np.zeros((1, 2))
    pr = a[:, 0] * a[:, 1]
    print("  %-12s uout %.2f uin %.2f | product > 0.1: %.3f | > 0.3: %.3f" % (k, np.median(a[:, 0]), np.median(a[:, 1]), (pr > 0.1).mean(), (pr > 0.3).mean()))
