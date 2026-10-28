import json, os
from nuscenes.utils.splits import create_splits_scenes
s = create_splits_scenes(); old = json.load(open("data/scenes_parts1_6.json"))
oldall = set(old["train"]) | set(old["val"]) if isinstance(old, dict) else set()
allv = set(s["val"]); allt = set(s["train"])
meta = json.load(open(os.path.expanduser("~/data/nuscenes_full/v1.0-trainval/scene.json")))
names = [m["name"] for m in meta]
new = [n for n in names if n not in oldall]
print("keys", list(old)[:5] if isinstance(old, dict) else type(old))
print("new scenes", len(new), "train", sum(n in allt for n in new), "val", sum(n in allv for n in new))
