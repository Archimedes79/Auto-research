"""scratch: the trainval folder's index.json was overwritten by the later mini report (10 scenes); write the real one (850 scenes) with the fixed write_index"""
import json
import os
import sys
sys.path.insert(0, ".")
import autogt
from autogt.cli import write_index
from autogt.suite import AutoGT, default_root, json_tables

gen = AutoGT(default_root("v1.0-mini"), "v1.0-mini", ss=4, light=True)         # only its signature / grid / ss are used
sd = os.path.join("data", "autogt", autogt.current_signature())
root = default_root("v1.0-trainval")
scenes, _ = json_tables(root, "v1.0-trainval")
names = sorted(s["name"] for s in scenes if os.path.exists(os.path.join(sd, f"_scene_{s['name']}.pkl")))
n_kf = len([f for f in os.listdir(sd) if f.endswith(".npz") and ".tmp" not in f])
print("scenes with a finished-sequence pickle:", len(names), "| keyframe files:", n_kf)
assert len(names) == 850 and n_kf == 34149
old = json.load(open(os.path.join(sd, "index.json")))
write_index(sd, gen, "v1.0-trainval", names, 1, n_kf, {})
new = json.load(open(os.path.join(sd, "index.json")))
print("before:", old["version"], len(old["scenes"]), old["keyframes"], "| after:", new["version"], len(new["scenes"]), new["keyframes"], "complete", new["complete"])
write_index(sd, gen, "v1.0-mini", old["scenes"], 1, 404, {})                    # a later mini run must now leave it as it is
again = json.load(open(os.path.join(sd, "index.json")))
assert again["version"] == "v1.0-trainval" and len(again["scenes"]) == 850 and again["keyframes"] == 34149, again
print("a mini run afterwards keeps:", again["version"], len(again["scenes"]), again["keyframes"])
