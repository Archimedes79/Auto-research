"""One-off check (2026-10-31): does the constant-velocity shift v * dt move the annotated box onto the object?
There is no lidar in this data copy, so the independent evidence is the raw RADAR positions (13 Hz per radar, five radars; the
radar Doppler entered the velocity, the radar POSITIONS did not). For every moving observed vehicle (>= 5 m/s, 5..45 m ahead)
and every radar sweep within +-0.45 s of the keyframe we count the radar points inside the box footprint (+1.0 m margin: radar
position noise) for three boxes:
  none   the annotation pose (no motion)
  ours   the annotation pose moved by v * (t_sweep - t_annotation), v = the GT's radar-corrected velocity
  wrong  the same with the opposite sign (control)
Run from bev_fusion: bash temp/checks/run_py_nice.sh temp/checks/radar_box_shift.py [stride] [version]
"""
import collections
import math
import os
import sys

import numpy as np
from pyquaternion import Quaternion

sys.path.insert(0, os.getcwd())
from autogt import AutoGT                                        # noqa: E402

stride = int(sys.argv[1]) if len(sys.argv) > 1 else 2
version = sys.argv[2] if len(sys.argv) > 2 else "v1.0-mini"
root = os.path.expanduser("~/data/nuscenes")
gen = AutoGT(root, version, boxes_at="annotation")
nusc = gen.nusc
MARGIN, VMIN, WIN = 1.0, 5.0, 0.45
BINS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, WIN)]


def inside(xy, centre, yaw, size):
    w, l, _ = size
    d = xy - centre
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    return int(((np.abs(lx) < l / 2 + MARGIN) & (np.abs(ly) < w / 2 + MARGIN)).sum())


res = collections.defaultdict(lambda: collections.defaultdict(list))      # (bin, sign of dt) -> variant -> counts
n_boxes = 0
for name in gen.scenes():
    sc = gen.scene(nusc.get("scene", next(s["token"] for s in nusc.scene if s["name"] == name))["token"])
    sc.radar_window = WIN
    for k, tok in enumerate(gen.keyframes(name)):
        if k % stride:
            continue
        gk = gen.keyframe(tok)
        Re = gk["meta"]["T_e2g"][:2, :2]
        sample = nusc.get("sample", tok)
        t_ann = sample["timestamp"] * 1e-6
        boxes = []
        for b in gk["meta"]["boxes"]:
            if b["group"] != "vehicle" or not b["observed"] or b["stationary"] or not b["vel_known"]:
                continue
            a = nusc.get("sample_annotation", b["token"])
            vw = Re @ b["vel"]
            if np.hypot(*vw) < VMIN or not (5 < b["center"][0] < 45):
                continue
            boxes.append((np.array(a["translation"]), Quaternion(a["rotation"]).yaw_pitch_roll[0], a["size"], vw))
        if not boxes:
            continue
        n_boxes += len(boxes)
        for fr in sc.radar(t_ann):
            dt = fr.t - t_ann
            bi = next(i for i, (lo, hi) in enumerate(BINS) if lo <= abs(dt) < hi + 1e-9)
            near = fr.xy
            for trans, yaw, size, vw in boxes:
                for var, sg in (("none", 0.0), ("ours", 1.0), ("wrong", -1.0)):
                    res[(bi, dt > 0)][var].append(inside(near, trans[:2] + sg * vw * dt, yaw, size))
    print(name, n_boxes, "boxes", flush=True)

print(f"\n{version} stride {stride}: {n_boxes} moving observed vehicles (>= {VMIN} m/s, 5..45 m); radar points inside the box + {MARGIN} m")
print("|dt| s       sign  n(box-frames)  none   ours  wrong | ours>none ours<none")
for (bi, pos), r in sorted(res.items()):
    n, o, w_ = np.array(r["none"]), np.array(r["ours"]), np.array(r["wrong"])
    lo, hi = BINS[bi]
    print(f"{lo:.2f}..{hi:.2f}   {'+' if pos else '-'}   {len(n):8d}   {n.mean():6.2f} {o.mean():6.2f} {w_.mean():6.2f} |  {np.mean(o > n):5.1%}    {np.mean(o < n):5.1%}")
for lo, hi in ((0.0, 0.1), (0.1, 0.45)):
    sel = [r for (bi, _), r in res.items() if lo <= BINS[bi][0] < hi]
    t = {v: np.concatenate([np.array(r[v]) for r in sel]) for v in ("none", "ours", "wrong")}
    print(f"|dt| in [{lo}, {hi}): points per box-frame none {t['none'].mean():.2f}  ours {t['ours'].mean():.2f}  wrong {t['wrong'].mean():.2f}  "
          f"(ours/none {t['ours'].sum() / max(t['none'].sum(), 1):.2f}, wrong/none {t['wrong'].sum() / max(t['none'].sum(), 1):.2f})")
