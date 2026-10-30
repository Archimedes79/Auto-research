"""t7 scratch: headline map / detection numbers (front FOV, dt 0) of stored runs: python temp/t7/ref_maps.py <run> ..."""
import json
import sys

KEYS = ["lane_divider/all/all/iou@0.5", "lane_divider/all/all/iou@1", "road_divider/all/all/iou@1",
        "lane_divider/0-20/iou@0.5", "lane_divider:zc/all/all/f1@0.5", "drivable_area/all/all/iou@0.5",
        "walkway/all/all/iou@0", "ped_crossing/all/all/iou@0"]

for r in sys.argv[1:]:
    try:
        m = json.load(open(f"runs/{r}/eval/metrics.json"))["metrics"]
    except FileNotFoundError:
        print(r, "no metrics")
        continue
    row = {k.split("/")[0] + "/" + k.split("/")[-1] + ("" if "/all/all/" in k else " " + k.split("/")[1]):
           round(m.get("seg_fov/seg/dt+0.00/" + k.replace("0-20/", "all/0-20/"), float("nan")), 3) for k in KEYS}
    row["mAP"] = round(m.get("det/dt+0.00/mean/all/mAP", float("nan")), 3)
    print(r, row)
