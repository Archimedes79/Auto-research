"""How many frames with cached GaussianCaR logits (lane project) lie in the official val scenes we evaluate on?
Throw-away check for the SOA-gap analysis (2026-10-30). CPU, one thread."""
import json
import os

from nuscenes.nuscenes import NuScenes

HOME = os.path.expanduser("~")
nusc = NuScenes(version="v1.0-trainval", dataroot=f"{HOME}/data/nuscenes_full", verbose=False)
ours = json.load(open("data/scenes_parts1_6.json"))
val_scenes = set(ours["val"])
print("our val scenes", len(val_scenes), "train scenes", len(ours["train"]))
for split in ("val", "train"):
    toks = json.load(open(f"{HOME}/exp_gcar_b123/feats/{split}_tokens.json"))
    n_in, scenes = 0, set()
    idx = []
    for i, t in enumerate(toks):
        s = nusc.get("scene", nusc.get("sample", t)["scene_token"])["name"]
        if s in val_scenes:
            n_in += 1
            scenes.add(s)
            idx.append(i)
    print(split, "tokens", len(toks), "in our val scenes", n_in, "scenes", len(scenes))
    if split == "val":
        json.dump(idx, open("temp/soa_gap/gcar_val_idx_in_our_val.json", "w"))
