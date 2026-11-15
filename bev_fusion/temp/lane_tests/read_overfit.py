"""scratch: print the lane-related numbers of the before / after evaluation of the overfit test (read from the card files)."""
import json
import sys

KEYS = ["seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1",
        "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1", "seg_fov/seg/dt+0.00/lane_divider:zc/all/all/f1@0.5",
        "seg_fov/seg/dt+0.00/road_divider:zc/all/all/f1@0.5", "seg_fov/seg/dt+0.00/centreline/all/all/iou@0.5",
        "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0.5", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0",
        "seg_fov/seg/dt+0.00/lane_divider/all/all/n_gt"]
runs = ["runs/overfit8_before", "runs/overfit8_s0"]
cards = [json.load(open(r + "/eval/card_train.json")) for r in runs]
print("frames", [c["frames"] for c in cards])
for k in KEYS:
    print(f"{k.replace('seg_fov/seg/dt+0.00/', ''):48s}", *[f"{c['metrics'].get(k, float('nan')):10.4f}" for c in cards])
extra = sorted({k for c in cards for k in c["metrics"] if "lane_divider" in k and "dt+0.00" in k and "/all/all/" in k})
print(len(extra), "lane_divider keys at dt 0")
for k in extra:
    print(f"{k.replace('seg_fov/seg/dt+0.00/', ''):48s}", *[f"{c['metrics'].get(k, float('nan')):10.4f}" for c in cards])
