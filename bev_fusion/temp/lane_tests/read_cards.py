"""scratch: lane-line numbers (val, fov, dt 0) of stored run cards, side by side. python temp/lane_tests/read_cards.py run1 run2 ..."""
import json
import sys

runs = sys.argv[1:]
cards = []
for r in runs:
    for fn in ("card.json", "card_val.json"):
        try:
            cards.append(json.load(open(f"runs/{r}/eval/{fn}")))
            break
        except FileNotFoundError:
            continue
    else:
        cards.append({"metrics": {}, "frames": "-"})
print("frames", [c.get("frames") for c in cards])
keys = sorted({k for c in cards for k in c["metrics"] if "seg_fov/seg/dt+0.00/" in k and "/all/all/" in k and ("lane_divider" in k or "road_divider" in k or "centreline" in k)})
for k in keys:
    if k.endswith("n_gt"):
        continue
    print(f"{k.replace('seg_fov/seg/dt+0.00/', ''):52s}", *[f"{c['metrics'].get(k, float('nan')):9.4f}" for c in cards])
