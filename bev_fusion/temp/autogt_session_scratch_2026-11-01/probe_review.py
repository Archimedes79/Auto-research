"""review probe (read-only): camera-time boxes on nuScenes mini, scene-0061 only, light devkit, CPU."""
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from autogt import AutoGT                       # noqa: E402
from autogt.grid import GridSpec                # noqa: E402

root = os.path.expanduser("~/data/nuscenes")
g = GridSpec()
cam = AutoGT(root, "v1.0-mini", light=True, scenes=["scene-0061"])
ann = AutoGT(root, "v1.0-mini", nusc=cam.nusc, boxes_at="annotation", light=True, scenes=["scene-0061"])
toks = cam.keyframes("scene-0061")
print("keyframes", len(toks))
n_unknown = n_moved = n_stat = n_zero_shift = 0
unk_bad = 0
for tok in toks:
    k = cam.keyframe(tok)
    a = ann.keyframe(tok)
    for b, c in zip(k["meta"]["boxes"], a["meta"]["boxes"]):
        if b["stationary"]:
            n_stat += 1
        elif not b["vel_known"]:
            n_unknown += 1
            if b["shift_s"] != 0 or not np.allclose(b["center"], c["center"]):
                unk_bad += 1
        else:
            n_moved += 1
            if np.allclose(b["shift"], 0):
                n_zero_shift += 1
print("stationary", n_stat, "moved(known,moving)", n_moved, "of which zero shift", n_zero_shift, "unknown", n_unknown, "unknown bad", unk_bad)

tok = toks[8]
k = cam.keyframe(tok)
sc = cam.nusc.get("sample", tok)["scene_token"]
fixed = cam.stationary_instances(sc)
alias = [np.shares_memory(b["world_ann"], fixed[b["instance"]][0]) for b in k["meta"]["boxes"] if b["stationary"]]
alias_w = [np.shares_memory(b["world"], fixed[b["instance"]][0]) for b in k["meta"]["boxes"] if b["stationary"]]
print("stationary boxes:", len(alias), "world_ann aliases the cache:", sum(alias), "world aliases the cache:", sum(alias_w))

track = cam.camera_track(tok)
sd0 = cam.cam(tok)
i = [s["token"] for s in track].index(sd0["token"])
smp = cam.nusc.get("sample", tok)
for j in (1, 3, -3):
    sd = track[i + j]
    s = cam.keyframe(tok, sd=sd)
    t_exp = (sd["timestamp"] - smp["timestamp"]) * 1e-6
    mv = [b for b in s["meta"]["boxes"] if b["shift_s"] != 0]
    shifts = {round(b["shift_s"], 4) for b in mv}
    mx = max((np.hypot(*b["shift"]) for b in mv), default=0)
    print(f"sd offset {j:+d}: expected t {t_exp:+.4f}, shift_s set {shifts}, max shift {mx:.2f} m, annotation_timestamp ok",
          s["meta"]["annotation_timestamp"] == smp["timestamp"])

nxt = smp["next"]
kn = ann.keyframe(nxt)
ka = ann.keyframe(tok)
j = min(range(len(track)), key=lambda q: abs(track[q]["timestamp"] * 1e-6 - (smp["timestamp"] * 1e-6 + 0.25)))
sd = track[j]
ks = cam.keyframe(tok, sd=sd)
t = (sd["timestamp"] - smp["timestamp"]) * 1e-6
tn = (cam.nusc.get("sample", nxt)["timestamp"] - smp["timestamp"]) * 1e-6
errs = []
by_next = {b["instance"]: b for b in kn["meta"]["boxes"]}
for b, c in zip(ks["meta"]["boxes"], ka["meta"]["boxes"]):
    o = by_next.get(b["instance"])
    if o is None or b["stationary"] or not b["vel_known"] or abs(t) < 1e-3:
        continue
    interp = c["world"] + (o["world"] - c["world"]) * (t / tn)
    errs.append(np.hypot(*(b["world"][:2] - interp[:2])))
print(f"sweep at {t:+.3f}s: extrapolated vs interpolated centre, n {len(errs)}, median {np.median(errs) if errs else float('nan'):.3f} m, max {np.max(errs) if errs else float('nan'):.3f} m")
