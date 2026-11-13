"""E276 pre-experiment P4 (diagnostic, CPU): how good is the MAP EXTENSION rule of the tier-1 label (the straightest successor / predecessor, used where the car did not drive far enough: in about half of
the frames)? Method: take stored l4 frames where the car really drove >= 54 m ahead and >= 24 m behind (the tier-1 label is then the DRIVEN path, no extension: the truth), cut the driven path to the first
X m ahead (X = 10, 20, 30) resp. Y m behind, label again with the extension rule (`lane_tiers.tier_pieces` of the copy temp/gt_l4, the parameters stored in the graph) and compare the tier-1 points.
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_extension_accuracy.py <store folder> [n_frames]
"""
import copy
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.abspath(os.path.join(ROOT, "..", "temp", "gt_l4")))
from autogt import lane_tiers as LT  # noqa: E402

store = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
files = sorted(glob.glob(os.path.join(store, "*.pkl.gz")))
rng = np.random.RandomState(1)
files = [files[i] for i in sorted(rng.choice(len(files), min(len(files), 3 * n), replace=False))]


def cut(path, ahead=None, back=None):
    """the driven path cut to `ahead` m in front of the keyframe position (0, 0) and `back` m behind it (the path is oldest first; the keyframe position is the point nearest to the origin)"""
    P = np.asarray(path["pts"], float)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    k0 = int(np.argmin(np.linalg.norm(P, axis=1)))
    keep = np.ones(len(P), bool)
    if ahead is not None:
        keep &= arc <= arc[k0] + ahead
    if back is not None:
        keep &= arc >= arc[k0] - back
    q = P[keep]
    return dict(pts=q.astype(np.float32), ahead_m=float(min(ahead if ahead is not None else path["ahead_m"], path["ahead_m"])), back_m=float(min(back if back is not None else path["back_m"], path["back_m"])))


res = {}
frames = 0
for fpath in files:
    if frames >= n:
        break
    g = pickle.load(gzip.open(fpath, "rb"))
    if not g["pieces"] or "ego_path" not in g:
        continue
    pa = g["ego_path"]
    if pa["ahead_m"] < 54.0 or pa["back_m"] < 24.0 or g["tier_info"]["extension_m"] > 0 or g["tier_info"]["back_extension_m"] > 0:
        continue
    frames += 1
    p = dict(g["params"])
    truth = [np.asarray(q["tier"]) == 1 for q in g["pieces"]]
    for name, kw in (("ahead 10 m", dict(ahead=10.0)), ("ahead 20 m", dict(ahead=20.0)), ("ahead 30 m", dict(ahead=30.0)), ("behind 5 m", dict(back=5.0)), ("both 10 / 5 m", dict(ahead=10.0, back=5.0))):
        h = copy.deepcopy(g)
        LT.tier_pieces(h, cut(pa, **kw), p)
        got = [np.asarray(q["tier"]) == 1 for q in h["pieces"]]
        tp = sum(float((a & b).sum()) for a, b in zip(truth, got)) * 0.5
        gt_len = sum(float(a.sum()) for a in truth) * 0.5
        gt_got = sum(float(b.sum()) for b in got) * 0.5
        r = res.setdefault(name, dict(tp=0.0, truth=0.0, got=0.0, bad=0, frames=0, worst=[]))
        r["tp"] += tp; r["truth"] += gt_len; r["got"] += gt_got; r["frames"] += 1
        rec = tp / max(gt_len, 1e-9)
        r["bad"] += int(rec < 0.8)
print(f"{frames} stored frames with a fully driven tier-1 path (>= 54 m ahead, >= 24 m behind), the path cut and the tier-1 label redone by the map extension rule")
print("| cut path | tier-1 recall of the extension rule | precision | frames with recall < 0.8 |")
print("|---|---|---|---|")
out = {}
for name, r in res.items():
    out[name] = dict(recall=r["tp"] / r["truth"], precision=r["tp"] / max(r["got"], 1e-9), frames=r["frames"], frames_recall_below_0_8=r["bad"] / r["frames"])
    print(f"| {name} | {out[name]['recall']:.3f} | {out[name]['precision']:.3f} | {100 * out[name]['frames_recall_below_0_8']:.1f} % |")
json.dump(out, open(os.path.join(ROOT, "temp", "e276", "extension_accuracy.json"), "w"), indent=1)
