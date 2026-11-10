"""Integrity of the BEVFusion feature cache (plan 2.8 step 0): every scene of the DualBEV cache has a shard with the same tokens and shape (n, 198, 32, 88) float16;
two frames per scene are read: finite, probabilities sum to 1, context not all zero. Run: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_bevfusion_cache.py"""
import glob
import os

import numpy as np

src, out = "data/cache/cam_feats", "data/cache/cam_feats_bevfusion"
n_src = n_out = bad = 0
stats = []
for f in sorted(glob.glob(os.path.join(src, "*.index.npz"))):
    scene = os.path.basename(f)[:-len(".index.npz")]
    t_src = np.load(f)["tokens"]
    fo = os.path.join(out, scene + ".index.npz")
    if not os.path.exists(fo):
        print("missing", scene)
        bad += 1
        continue
    t_out = np.load(fo)["tokens"]
    arr = np.load(os.path.join(out, scene + ".npy"), mmap_mode="r")
    ok = set(t_src.tolist()) <= set(t_out.tolist()) and arr.shape == (len(t_out), 198, 32, 88) and arr.dtype == np.float16
    for r in (0, len(t_out) - 1):
        a = np.asarray(arr[r]).astype(np.float32)
        ok &= bool(np.isfinite(a).all() and abs(a[:118].sum(0) - 1).max() < 5e-3 and np.abs(a[118:]).max() > 0)
        stats.append((a[:118].max(0).mean(), (a[:118] * (1.0 + 0.5 * np.arange(118)).reshape(118, 1, 1)).sum(0).mean()))
    n_src += len(t_src)
    n_out += len(t_out)
    bad += int(not ok)
    if not ok:
        print("PROBLEM", scene, arr.shape, len(t_out))
st = np.mean(stats, 0)
print(f"{n_out} frames in {n_src and len(glob.glob(os.path.join(out, '*.npy')))} shards (source {n_src}); shards with a problem: {bad}; mean peak probability {st[0]:.3f}, mean expected depth {st[1]:.1f} m")
