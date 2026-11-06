"""One-off check of the camera-time boxes (2026-10-31), mini, run from bev_fusion:
 1 boxes_at="annotation" reproduces the stored GT of the previous generator (signature 7fe1149566) layer by layer
 2 boxes_at="camera" differs from it only in the box-dependent layers, and by how much
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from autogt import AutoGT                                  # noqa: E402
from autogt.cli import load_keyframe                       # noqa: E402

OLD = "data/autogt/7fe1149566"
root = os.path.expanduser("~/data/nuscenes")
ann = AutoGT(root, "v1.0-mini", boxes_at="annotation")
cam = AutoGT(root, "v1.0-mini", nusc=ann.nusc)
files = sorted(f[:-4] for f in os.listdir(OLD) if f.endswith(".npz") and f[:-4] in ann.nusc._token2ind["sample"])   # mini tokens
files = files[::8]
box_layers = {"vehicle", "pedestrian", "two_wheeler", "instance", "ignore", "velocity", "flow", "flow_weight", "flow_sigma", "occluded"}
bad, diffs, n = [], {}, 0
for tok in files:
    old = load_keyframe(OLD, tok)
    a, c = ann.keyframe(tok), cam.keyframe(tok)
    n += 1
    q = lambda x: x.astype(np.float16).astype(np.float32) if x.dtype == np.float32 else x   # noqa: E731  (the cache stores float16)
    for k in old:
        if k == "meta":
            continue
        if not np.array_equal(old[k], q(a[k])):
            bad.append((tok[:8], k, float(np.abs(old[k] - q(a[k])).max())))
        if not np.array_equal(old[k], q(c[k])):
            diffs.setdefault(k, []).append(float(np.abs(old[k] - q(c[k])).mean()))
print(f"{n} keyframes; annotation mode vs the old stored GT: {len(bad)} differing layers {bad[:5]}")
print("camera mode: layers that differ from the old GT (keyframes affected / mean abs difference):")
for k, v in sorted(diffs.items()):
    print(f"  {k:14s} {len(v):3d}/{n}  {np.mean(v):.5f}  {'(box layer)' if k in box_layers else '(NOT a box layer!)'}")
assert not bad, "annotation mode must reproduce the previous generator"
assert all(k in box_layers for k in diffs), [k for k in diffs if k not in box_layers]
print("ok")
