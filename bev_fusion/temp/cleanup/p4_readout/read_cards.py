"""scratch: the numbers of the three P4 read-out variants from their metric cards (never typed by hand)"""
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
FILES = {"a as is (0.5 s)": "a/eval/card.json", "b zero_radar": "b/eval/card_ctl_zero_radar.json", "c newest sweep (0.09 s)": "c/eval/card.json"}
cards = {k: json.load(open(os.path.join(BASE, v))) for k, v in FILES.items()}
first = next(iter(cards.values()))
m0 = first.get("metrics", first)
print("top keys:", list(first)[:12])
keys = [k for k in m0 if re.search(r"detf/dt\+0\.00/(mean|vehicle|pedestrian)/all/(mAP|AP@1|AP@2)$", k)]
keys += [k for k in m0 if re.search(r"^det/dt\+0\.00/mean/all/mAP$", k)]
keys += [k for k in m0 if re.search(r"seg(\.fov)?/dt\+0\.00/(drivable_area|lane_divider|road_divider)/all/all/iou@(0|0\.5)$", k)]
keys += [k for k in m0 if re.search(r"vel/dt\+0\.00/centre/field/moving/epe$", k)]
print(f"{'metric':64s}" + "".join(f"{n:>26s}" for n in cards))
for k in dict.fromkeys(keys):
    vals = [(c.get("metrics", c)).get(k) for c in cards.values()]
    print(f"{k:64s}" + "".join(f"{v:26.4f}" if isinstance(v, (int, float)) else f"{str(v):>26s}" for v in vals))
print("n / provenance:", {n: (c.get("n"), (c.get("provenance") or {}).get("evaluator", {}).get("dirty"), (c.get("protocol") or {}).get("control")) for n, c in cards.items()})
