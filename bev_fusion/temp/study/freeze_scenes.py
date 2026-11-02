"""One-off: freeze the scene names of the current train / val sets (parts 1-6: 428 / 82 scenes) so later downloads of parts 7-10
do not change what existing configs select. Writes data/scenes_parts1_6.json and patches training/configs/t2_*.json and
runs/t2_*/config.json (train_scenes / val_scenes)."""
import glob, json, os, sys
sys.path.insert(0, "training")
from nuscenes.nuscenes import NuScenes
from nuscenes_dataset import available_scenes
nusc = NuScenes("v1.0-trainval", dataroot=os.path.expanduser("~/data/nuscenes_full"), verbose=False)
out = {s: sorted(nusc.get("scene", t)["name"] for t in available_scenes(nusc, s)) for s in ("train", "val")}
print({k: len(v) for k, v in out.items()})
assert (len(out["train"]), len(out["val"])) == (428, 82), "scene counts changed already"
json.dump(out, open("data/scenes_parts1_6.json", "w"))
for p in glob.glob("training/configs/t2_*.json") + glob.glob("runs/t2_*/config.json"):
    c = json.load(open(p))
    c["train_scenes"], c["val_scenes"] = out["train"], out["val"]
    json.dump(c, open(p, "w"), indent=1)
    print("patched", p)
