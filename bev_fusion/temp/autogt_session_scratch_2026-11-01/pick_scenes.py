"""scratch: candidate trainval sequences for the crosscheck videos - one per city, with weather / night variety, from the json tables."""
import json
import os
import sys

sys.path.insert(0, ".")
from autogt.suite import default_root

root = os.path.join(default_root("v1.0-trainval"), "v1.0-trainval")
scenes = json.load(open(os.path.join(root, "scene.json")))
logs = {r["token"]: r["location"] for r in json.load(open(os.path.join(root, "log.json")))}
by = {}
for s in scenes:
    by.setdefault(logs[s["log_token"]], []).append(s)
for loc, ss in sorted(by.items()):
    print(loc, len(ss))
    for key in ("rain", "night", "construction", "bus", "pedestrian", "intersection", "parked"):
        hit = [s for s in ss if key in s["description"].lower()]
        print("   ", key, len(hit), [(s["name"], s["description"][:60]) for s in hit[:3]])
