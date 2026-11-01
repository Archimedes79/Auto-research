"""scratch: the five radar mounting poses (x, y, yaw) in the ego frame over all logs of nuScenes trainval (json tables only)"""
import json
import math
import os
from collections import defaultdict

root = os.path.expanduser("~/data/nuscenes_full/v1.0-trainval")
sensor = {s["token"]: s["channel"] for s in json.load(open(os.path.join(root, "sensor.json")))}
cal = json.load(open(os.path.join(root, "calibrated_sensor.json")))
by = defaultdict(set)
for c in cal:
    ch = sensor[c["sensor_token"]]
    if not ch.startswith("RADAR"):
        continue
    w, x, y, z = c["rotation"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    by[ch].add((round(c["translation"][0], 3), round(c["translation"][1], 3), round(math.degrees(yaw), 1)))
for ch, v in sorted(by.items()):
    print(ch, len(v), "distinct poses", sorted(v)[:6])
