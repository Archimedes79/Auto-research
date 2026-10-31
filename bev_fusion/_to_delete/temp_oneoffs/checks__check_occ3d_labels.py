"""Verify the downloaded Occ3D-nuScenes labels (D:\\Auto-research\\NuScenes\\occ3d): layout, counts, one frame, overlap with our 510 scenes.

Checks: annotations.json structure and the number of scenes / frames; gts/<scene_name>/<sample_token>/labels.npz present for every
frame of annotations.json; one labels.npz (shapes, dtypes, class ids and their shares); the share of our 510 scenes (the scenes with
camera + radar on D:) found by scene name and of their keyframes by sample token (nuScenes metadata of the trainval folder).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_occ3d_labels.py
"""
import json
import os
import time

import numpy as np

root = "/mnt/d/Auto-research/NuScenes/occ3d"
t0 = time.time()
ann = json.load(open(os.path.join(root, "annotations.json")))
print(f"annotations.json: type {type(ann).__name__}, top-level keys {list(ann)[:6]} ({time.time() - t0:.0f} s)")
scenes = ann["scene_infos"] if "scene_infos" in ann else ann
names = list(scenes)
print(f"scenes: {len(names)}, first {names[:3]}")
frames = {}
for s in names:
    v = scenes[s]
    toks = list(v) if isinstance(v, dict) else [f.get("token", f.get("sample_token")) for f in v]
    frames[s] = toks
n_frames = sum(len(v) for v in frames.values())
print(f"frames in annotations.json: {n_frames}; example frame entry keys: " +
      str(list(scenes[names[0]][list(scenes[names[0]])[0]].keys())[:8] if isinstance(scenes[names[0]], dict) else "list"))
gts = os.path.join(root, "gts")
have_scenes = sorted(os.listdir(gts))
print(f"gts/: {len(have_scenes)} scene folders ({time.time() - t0:.0f} s)")
missing, n_have = 0, 0
for s in names:
    d = os.path.join(gts, s)
    present = set(os.listdir(d)) if os.path.isdir(d) else set()
    n_have += len(present)
    missing += sum(1 for t in frames[s] if t not in present)
print(f"frame folders present: {n_have}; frames of annotations.json without a folder: {missing}")
s0 = names[0]
t = frames[s0][0]
z = np.load(os.path.join(gts, s0, t, "labels.npz"))
print(f"one frame {s0}/{t}: " + ", ".join(f"{k} {z[k].shape} {z[k].dtype}" for k in z.files))
sem = z["semantics"]
ids, cnt = np.unique(sem, return_counts=True)
print("class ids and shares in this frame: " + ", ".join(f"{i}: {100 * c / sem.size:.1f} %" for i, c in zip(ids, cnt)))
print(f"mask_lidar visible share {100 * z['mask_lidar'].mean():.1f} %, mask_camera visible share {100 * z['mask_camera'].mean():.1f} %")
# our scenes: nuScenes metadata on D: (scene name -> scene token; sample tokens)
meta = "/mnt/d/Auto-research/NuScenes/nuscenes_full/v1.0-trainval"
sc = json.load(open(os.path.join(meta, "scene.json")))
sm = json.load(open(os.path.join(meta, "sample.json")))
cam = os.path.join("/mnt/d/Auto-research/NuScenes/nuscenes_full/samples/CAM_FRONT")
tok_by_scene = {}
for r in sm:
    tok_by_scene.setdefault(r["scene_token"], []).append(r["token"])
ours = [s for s in sc if os.path.exists(os.path.join(cam, ""))]   # all 850 in metadata; keyframes of our 510 are on disk (checked below)
on_disk = set(os.listdir(cam))
sd = json.load(open(os.path.join(meta, "sample_data.json")))
disk_tokens = {r["sample_token"] for r in sd if r["is_key_frame"] and r["filename"].startswith("samples/CAM_FRONT/")
               and os.path.basename(r["filename"]) in on_disk}
ours_names = [s["name"] for s in sc if set(tok_by_scene[s["token"]]) & disk_tokens]
print(f"our scenes (CAM_FRONT keyframes on D:): {len(ours_names)}")
inter = [n for n in ours_names if n in frames]
tok_ours = {t for s in sc if s["name"] in inter for t in tok_by_scene[s["token"]]}
tok_occ = {t for n in inter for t in frames[n]}
print(f"found in Occ3D by scene name: {len(inter)} of {len(ours_names)} scenes; keyframes of those scenes present in Occ3D: "
      f"{len(tok_ours & tok_occ)} of {len(tok_ours)}")
print(f"done ({time.time() - t0:.0f} s)")
