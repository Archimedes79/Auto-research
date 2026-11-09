"""One-off: official train / val scenes and CAM_FRONT keyframes present in the downloaded parts (for the S1 run plan).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/setup/count_split_scenes.py"""
import os

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

root = os.path.expanduser("~/data/nuscenes_full")
nusc = NuScenes("v1.0-trainval", root, verbose=False)
splits = create_splits_scenes()
by_name = {s["name"]: s for s in nusc.scene}
for split in ("train", "val"):
    n_sc, n_kf, locs = 0, 0, {}
    for name in splits[split]:
        sc = by_name[name]
        tok, kf, ok = sc["first_sample_token"], 0, True
        while tok:
            smp = nusc.get("sample", tok)
            sd = nusc.get("sample_data", smp["data"]["CAM_FRONT"])
            ok = ok and os.path.exists(os.path.join(root, sd["filename"]))
            kf += 1
            tok = smp["next"]
        if ok:
            n_sc += 1
            n_kf += kf
            loc = nusc.get("log", sc["log_token"])["location"]
            locs[loc] = locs.get(loc, 0) + 1
    print(f"{split}: {n_sc} of {len(splits[split])} scenes present, {n_kf} CAM_FRONT keyframes, by location {locs}")
