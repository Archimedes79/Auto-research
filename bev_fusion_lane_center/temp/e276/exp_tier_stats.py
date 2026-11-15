"""E276 pre-experiment P2 (diagnostic, CPU): what the l4 labels look like, from the stored graphs (no autogt import: the files are plain pickles). Answers prediction 2 of the card (shares of the GT lane
length per tier, visible shares) and the critic's findings 2 / 3 (own lane at the log start, the blind zone at the car).
    wsl env CUDA_VISIBLE_DEVICES= bash tools/env/_py14.sh exp_tier_stats.py <store folder> [n_frames] [--val]
`--val` restricts to the 873 VAL frames of data/splits.json (needs the VAL tokens to be stored).
"""
import glob
import gzip
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
store = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
files = sorted(glob.glob(os.path.join(store, "*.pkl.gz")))
if "--val" in sys.argv:
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    sys.path.insert(0, os.path.join(ROOT, "training"))
    import qgraph_eval as QE
    want = {f.token for f in QE.val_frames()}
    files = [f for f in files if os.path.basename(f)[:-7] in want]
elif n:
    rng = np.random.RandomState(0)
    files = [files[i] for i in sorted(rng.choice(len(files), min(n, len(files)), replace=False))]
L = np.zeros((3, 4))                      # per tier: metres, demanded-by-camera metres, occ metres, uncovered (not occ, not vis) metres
rng_bins = [0, 3, 5, 8, 12, 20, 35, 50, 70]
R = np.zeros((len(rng_bins) - 1, 3))      # tier-1 metres per range bin: total, camera-visible, occ
share1 = []; none1 = 0; back = []; ahead = []; ext = []; bext = []; frames = 0; empty = 0
for fpath in files:
    g = pickle.load(gzip.open(fpath, "rb"))
    if not g["pieces"]:
        empty += 1
        continue
    frames += 1
    tl = np.zeros(3)
    for p in g["pieces"]:
        t, v, o = np.asarray(p["tier"]), np.asarray(p["vis"]), np.asarray(p["occ"])
        for k in (1, 2, 3):
            m = t == k
            L[k - 1] += 0.5 * np.array([m.sum(), (m & v).sum(), (m & o).sum(), (m & ~v & ~o).sum()])
        m1 = t == 1
        if m1.any():
            d = np.linalg.norm(np.asarray(p["pts"], float)[m1], axis=1)
            b = np.digitize(d, rng_bins) - 1
            for i in range(len(rng_bins) - 1):
                s = b == i
                R[i] += 0.5 * np.array([s.sum(), (s & v[m1]).sum(), (s & o[m1]).sum()])
    ti = g["tier_info"]
    if ti["length_m"][1] == 0:
        none1 += 1
    back.append(ti["driven_back_m"]); ahead.append(ti["driven_ahead_m"]); ext.append(ti["extension_m"]); bext.append(ti["back_extension_m"])
    share1.append(ti["length_m"][1] + ti["length_m"][2])
tot = L[:, 0].sum()
out = dict(store=os.path.basename(store), frames=frames, empty_frames=empty, frames_without_tier1=none1,
           share_of_gt_length=dict(zip(("tier1", "tier2", "tier3"), (L[:, 0] / tot).tolist())), visible_by_camera=dict(zip(("tier1", "tier2", "tier3"), (L[:, 1] / np.maximum(L[:, 0], 1)).tolist())),
           hidden_by_box=dict(zip(("tier1", "tier2", "tier3"), (L[:, 2] / np.maximum(L[:, 0], 1)).tolist())), outside_every_camera=dict(zip(("tier1", "tier2", "tier3"), (L[:, 3] / np.maximum(L[:, 0], 1)).tolist())),
           metres_per_frame=dict(zip(("tier1", "tier2", "tier3"), (L[:, 0] / frames).tolist())), tier12_metres_per_frame=float(np.mean(share1)),
           driven_back_m_mean=float(np.mean(back)), driven_ahead_m_mean=float(np.mean(ahead)), frames_with_forward_extension=int(np.sum(np.array(ext) > 0)), frames_with_back_extension=int(np.sum(np.array(bext) > 0)),
           tier1_by_range=[dict(range_m=f"{rng_bins[i]}-{rng_bins[i + 1]}", metres_per_frame=R[i, 0] / frames, camera_visible=float(R[i, 1] / max(R[i, 0], 1e-9)), hidden_by_box=float(R[i, 2] / max(R[i, 0], 1e-9)))
                           for i in range(len(rng_bins) - 1)])
json.dump(out, open(os.path.join(ROOT, "temp", "e276", f"tier_stats_{os.path.basename(store)}{'_val' if '--val' in sys.argv else ''}.json"), "w"), indent=1)
print(f"{out['store']}: {frames} frames ({empty} without any lane), no tier-1 lane in {none1}")
print("| tier | share of GT lane length | metres per frame | visible to a camera | hidden by a box | outside every camera |")
print("|---|---|---|---|---|---|")
for i, k in enumerate(("tier1", "tier2", "tier3")):
    print(f"| {i + 1} | {100 * out['share_of_gt_length'][k]:.1f} % | {out['metres_per_frame'][k]:.1f} | {100 * out['visible_by_camera'][k]:.0f} % | {100 * out['hidden_by_box'][k]:.0f} % | {100 * out['outside_every_camera'][k]:.0f} % |")
print(f"driven behind / ahead (mean m): {out['driven_back_m_mean']:.1f} / {out['driven_ahead_m_mean']:.1f}; map extension ahead in {out['frames_with_forward_extension']}, behind in {out['frames_with_back_extension']} frames")
print("tier 1 by range from the car: | m | m per frame | camera visible | hidden by box |")
for r in out["tier1_by_range"]:
    print(f"|{r['range_m']}| {r['metres_per_frame']:.2f} | {100 * r['camera_visible']:.0f} % | {100 * r['hidden_by_box']:.0f} % |")
