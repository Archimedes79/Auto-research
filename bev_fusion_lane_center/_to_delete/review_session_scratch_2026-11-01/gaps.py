import json, os
root = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
S = {s["token"]: s for s in json.load(open(f"{root}/sample.json"))}
SD = json.load(open(f"{root}/sample_data.json"))
cam = {}
for d in SD:
    if d["is_key_frame"] and "CAM_FRONT/" in d["filename"] and d["filename"].startswith("samples/"):
        cam[d["sample_token"]] = d["timestamp"]
gaps = []
for t, s in S.items():
    if s["next"] and t in cam and s["next"] in cam:
        gaps.append((cam[s["next"]] - cam[t]) * 1e-6)
import collections
r = collections.Counter(round(g / 0.5) * 0.5 for g in gaps)
print(len(gaps), r, min(gaps), max(gaps))
print(sum(1 for g in gaps if g >= 0.75), "gaps >= 0.75 s")
SC = {s["token"]: s for s in json.load(open(f"{root}/scene.json"))}
for t, s in S.items():
    if s["next"] and t in cam and s["next"] in cam and (cam[s["next"]] - cam[t]) * 1e-6 >= 0.7:
        g = (cam[s["next"]] - cam[t]) * 1e-6
        print(repr(g), round(g / 0.5) * 0.5, SC[s["scene_token"]]["name"], t, "prev" if s["prev"] else "first", "next.next" if S[s["next"]]["next"] else "next-is-last")
from nuscenes.utils.splits import create_splits_scenes
sp = create_splits_scenes()
for t, s in S.items():
    if s["next"] and t in cam and s["next"] in cam and (cam[s["next"]] - cam[t]) * 1e-6 >= 0.7:
        n = SC[s["scene_token"]]["name"]; print(n, "train" if n in sp["train"] else "val" if n in sp["val"] else "other")
fz = json.load(open("data/scenes_parts1_6.json")); print({k: n in v for k, v in fz.items()} if isinstance(fz, dict) else type(fz))
