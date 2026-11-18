"""E379 step-0 measurement (scratch, CPU): the competition structure of the champion's delivered LONG (junction) links, and what a destination-side competition at the decode would do to them. Frames: the VAL dump of a champion head
(runs/<run>/token_val.pkl, default e373_cb_s0; the D115 cell: long link threshold 0.2, both ends kept p_none < 0.6). Classes as `anatomy_long.py` (tp = a GT long link between the assigned tokens; dup_of_true = a link between tokens near the two ends of a GT link;
wrong_pair = both ends near GT nodes but no GT link between them; false_end = an end near no GT node). Filters (decode-side simulations, no retraining): keep only the highest-p incoming long link of every destination token ('per destination top-1'),
only the highest-p outgoing of every source ('per source top-1'), both ('mutual'); with the top-2 variants. Reports links, tp, precision (tp / links) and recall (tp / GT long links) of the long class per filter.
usage: python conflict_long.py [run]"""
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

R = os.environ["BFLC_ROOT"]
for sub in ("training/data", "training/grid", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(R, sub)))
import lane_tok_data as LT  # noqa: E402
import lane_tokcache as LC  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "e373_cb_s0"
seed = int(run[-1])
THR_LONG, KEEP = 0.2, 0.6
RAD, COS = 1.5, 0.5
fr = pickle.load(open(os.path.join(R, "runs", run, "token_val.pkl"), "rb"))
frames = fr["frames"] if isinstance(fr, dict) else fr
c = LC.TokCache(os.path.join(R, f"runs/e365_cache/be292_a_{seed}_K512_9ad1d4ce30_v1/val"))
ds = LT.TokData("val", res=2.0, feats=False)
tok2i = {t: i for i, t in enumerate(c.tokens)}
FILTERS = ("none", "dst1", "dst2", "src1", "src2", "mutual1", "mutual2")
acc = {f: dict(n=0, tp=0, dup=0, wrong=0, fend=0) for f in FILTERS}
n_gt_long = 0
contested = defaultdict(int)          # wrong / dup links whose destination (source) also carries a tp link of the same frame
tot_cls = defaultdict(int)
for f in frames:
    i = tok2i[f["token"]]
    t = ds.table(i)
    gpos, gdir = t["pos"], t["dirv"]
    pos, dirv = np.array(c.f["pos"][i], float), np.array(c.f["dirv"][i], float)
    gid = np.array(c.f["gt_id"][i])
    cd = f["cand"]
    src, dst, p, lg = (np.asarray(cd[k]) for k in ("src", "dst", "p", "long"))
    pnone = np.asarray(f["pnone"], float)
    sel = lg.astype(bool) & (p >= THR_LONG) & (pnone[src] < KEEP) & (pnone[dst] < KEEP)
    gl = set((int(a), int(b)) for a, b, l in zip(t["src"], t["dst"], t["long"]) if l)
    n_gt_long += len(gl)
    nearall = (np.linalg.norm(pos[:, None] - gpos[None], axis=2) <= RAD) & (dirv @ gdir.T >= COS)
    L = []
    for s_, d_, p_ in zip(src[sel], dst[sel], p[sel]):
        a, b = int(gid[s_]), int(gid[d_])
        if a >= 0 and b >= 0 and (a, b) in gl:
            k = "tp"
        else:
            ns, nd = np.flatnonzero(nearall[s_]), np.flatnonzero(nearall[d_])
            if len(ns) == 0 or len(nd) == 0:
                k = "fend"
            elif any((int(x), int(y)) in gl for x in ns for y in nd):
                k = "dup"
            else:
                k = "wrong"
        L.append((int(s_), int(d_), float(p_), k))
    by_d, by_s = defaultdict(list), defaultdict(list)
    for j, (s_, d_, p_, k) in enumerate(L):
        by_d[d_].append(j)
        by_s[s_].append(j)
    tp_d = {d_ for (s_, d_, p_, k) in L if k == "tp"}
    tp_s = {s_ for (s_, d_, p_, k) in L if k == "tp"}
    for (s_, d_, p_, k) in L:
        tot_cls[k] += 1
        if k != "tp":
            contested[k + "_dst_has_tp"] += d_ in tp_d
            contested[k + "_src_has_tp"] += s_ in tp_s
            contested[k + "_dst_shared"] += len(by_d[d_]) > 1
            contested[k + "_src_shared"] += len(by_s[s_]) > 1

    def rank_ok(groups, j, top):
        g = sorted(groups, key=lambda x: -L[x][2])
        return j in g[:top]

    for j, (s_, d_, p_, k) in enumerate(L):
        for name in FILTERS:
            top = int(name[-1]) if name[-1].isdigit() else 0
            if name == "none":
                keep = True
            elif name.startswith("dst"):
                keep = rank_ok(by_d[d_], j, top)
            elif name.startswith("src"):
                keep = rank_ok(by_s[s_], j, top)
            else:
                keep = rank_ok(by_d[d_], j, top) and rank_ok(by_s[s_], j, top)
            if keep:
                a = acc[name]
                a["n"] += 1
                a[k if k != "fend" else "fend"] += 1
print(run, "VAL frames", len(frames), "| GT long links", n_gt_long, "| delivered long links (p >= 0.2, both ends p_none < 0.6):", acc["none"]["n"])
print("classes of the delivered links:", {k: v for k, v in tot_cls.items()})
print("non-tp links by competition: of each class, the share whose DESTINATION carries a tp link / is shared with another delivered link; the same for the SOURCE")
for k in ("dup", "wrong", "fend"):
    n = max(tot_cls[k], 1)
    print("  %-6s n %5d | dst has tp %.3f | dst shared %.3f | src has tp %.3f | src shared %.3f" % (k, tot_cls[k], contested[k + "_dst_has_tp"] / n, contested[k + "_dst_shared"] / n, contested[k + "_src_has_tp"] / n, contested[k + "_src_shared"] / n))
print("decode-side competition filters on the delivered long links (precision = tp / links, recall = tp / GT long links):")
for name in FILTERS:
    a = acc[name]
    print("  %-8s links %6d tp %5d dup %5d wrong %5d fend %5d | precision %.3f recall %.3f" % (name, a["n"], a["tp"], a["dup"], a["wrong"], a["fend"], a["tp"] / max(a["n"], 1), a["tp"] / n_gt_long))
