"""One-off (2026-10-25): after the camera download, check completeness of the 12 Hz camera sweeps.

For every camera sample_data of the metadata: a scene counts as "on disk" if its CAM_FRONT keyframes exist. Every camera
sweep of an on-disk scene must exist (and be > 0 bytes); sweeps of the other scenes (parts 7-10) must be absent.
Run: wsl python3 datasets/NuScenes/tools/history/check_camera_sweeps.py
"""
import collections
import json
import os

root = os.path.expanduser("~/data/nuscenes_full")
meta = os.path.join(root, "v1.0-trainval")
sd = json.load(open(os.path.join(meta, "sample_data.json")))
samples = {s["token"]: s["scene_token"] for s in json.load(open(os.path.join(meta, "sample.json")))}
on_disk = set()
for r in sd:
    if r["is_key_frame"] and r["filename"].startswith("samples/CAM_FRONT/") and os.path.exists(os.path.join(root, r["filename"])):
        on_disk.add(samples[r["sample_token"]])
stat = collections.Counter()
missing = []
for r in sd:
    f = r["filename"]
    if not f.startswith("sweeps/CAM_"):
        continue
    cam = f.split("/")[1]
    scene = samples[r["sample_token"]]
    p = os.path.join(root, f)
    exists = os.path.exists(p) and os.path.getsize(p) > 0
    if scene in on_disk:
        stat[(cam, "expected")] += 1
        if exists:
            stat[(cam, "present")] += 1
        else:
            missing.append(f)
    elif exists:
        stat[(cam, "unexpected")] += 1
print(f"scenes with keyframes on disk: {len(on_disk)}")
for cam in sorted({c for c, _ in stat}):
    print(f"{cam}: expected {stat[(cam, 'expected')]}, present {stat[(cam, 'present')]}, unexpected {stat[(cam, 'unexpected')]}")
print(f"missing: {len(missing)}", missing[:5])
print("SWEEPS COMPLETE" if not missing else "SWEEPS INCOMPLETE")
