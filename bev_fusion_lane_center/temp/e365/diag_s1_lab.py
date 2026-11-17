"""E365 S1 diagnosis (scratch, IN-SAMPLE on VAL): where does the recall go: kept nodes, link recall on the token-level targets, type accuracy"""
import os, sys, pickle
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("training/data", "training/grid", "evaluation"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import lane_ptok_data as LP
S = int(sys.argv[1])
d = pickle.load(open(os.path.join(ROOT, f"runs/e365_s1_s{S}/token_val.pkl"), "rb"))
fr = d["frames"]
print("rec keys", list(fr[0]["rec"].keys()), {k: getattr(v, "shape", None) for k, v in fr[0]["rec"].items()})
ds, dl = LP.loader("val", seed=S, bs=8, workers=0, ident="e292_a")
i = 0
agg = dict(assigned=0, assigned_kept=0, kept=0, kept_unassigned=0, valid=0, ytype_hit=0, ytype_n=0, gt_pairs=0, gt_pairs_found=0, gt_pairs_src_kept=0, rec_links=0, rec_hit=0, src_with_gt=0, src_with_link=0, fp_none=0)
for b in dl:
    B = b["valid"].shape[0]
    for k in range(B):
        f = fr[i]; i += 1
        assert f["token"] == b["token"][k], (f["token"], b["token"][k])
        valid = b["valid"][k].numpy(); yt = b["ytype"][k].numpy(); y = b["y"][k].numpy() > 0
        keep = f["keep"].astype(bool) & valid
        ass = (yt > 0) & valid
        agg["valid"] += valid.sum(); agg["assigned"] += ass.sum(); agg["assigned_kept"] += (ass & keep).sum(); agg["kept"] += keep.sum(); agg["kept_unassigned"] += (keep & ~ass).sum()
        agg["ytype_n"] += ass.sum(); agg["ytype_hit"] += (f["ptype"].argmax(-1)[ass] == yt[ass]).sum()
        r = f["rec"]
        L = np.zeros_like(y)
        for key in r:
            pass
        src = np.asarray(r.get("src", [])); dst = np.asarray(r.get("dst", []))
        if len(src): L[src, dst] = True
        agg["gt_pairs"] += y.sum(); agg["gt_pairs_found"] += (y & L).sum()
        agg["gt_pairs_src_kept"] += (y & keep[:, None] & keep[None, :]).sum()
        agg["rec_links"] += L.sum(); agg["rec_hit"] += (L & y).sum()
        sg = y.any(1) & ass; agg["src_with_gt"] += sg.sum(); agg["src_with_link"] += (sg & L.any(1)).sum()
n = i
print("frames", n)
a = {k: float(v) / n for k, v in agg.items()}
print({k: round(v, 2) for k, v in a.items()})
print("node: assigned kept recall %.3f | kept tokens unassigned share %.3f | type acc on assigned %.3f" % (agg["assigned_kept"] / agg["assigned"], agg["kept_unassigned"] / agg["kept"], agg["ytype_hit"] / agg["ytype_n"]))
print("link: share of GT target pairs with both ends kept %.3f | pair recall (rec links hit y) %.3f | link precision vs y %.3f | sources with GT successor that got a link %.3f" % (agg["gt_pairs_src_kept"] / agg["gt_pairs"], agg["gt_pairs_found"] / agg["gt_pairs"], agg["rec_hit"] / max(agg["rec_links"], 1), agg["src_with_link"] / agg["src_with_gt"]))
