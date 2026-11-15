"""One-off: C: is full. The feature-cache shards of the 340 scenes of parts 7-10 (not used by any study run: scene lists are frozen) go to D:
(copy, size check, then remove from C:). Move back before a run that uses all 700 train scenes:  python temp/study/move_new_feats.py back"""
import json
import os
import shutil
import sys

src = "data/cache/cam_feats"
dst = "/mnt/d/Auto-research/bev_fusion_cache_parts7_10/cam_feats"
old = json.load(open("data/scenes_parts1_6.json"))
keep = set(old["train"]) | set(old["val"])
back = len(sys.argv) > 1 and sys.argv[1] == "back"
a, b = (dst, src) if back else (src, dst)
os.makedirs(b, exist_ok=True)
n = tot = 0
for f in sorted(os.listdir(a)):
    scene = f.split(".")[0]
    if (scene in keep) and not back:
        continue
    if back and scene in keep:
        continue
    s, d = os.path.join(a, f), os.path.join(b, f)
    shutil.copy2(s, d)
    assert os.path.getsize(s) == os.path.getsize(d), f
    os.remove(s)
    n += 1
    tot += os.path.getsize(d)
print(f"{'back' if back else 'moved'} {n} files, {tot / 1e9:.1f} GB: {a} -> {b}")
