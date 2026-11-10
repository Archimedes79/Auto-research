"""One-off (t3 try 1): headline metrics of the sanity controls next to the normal evaluation of the same run."""
import glob
import json
import os
K = [("mAP", "det/dt+0.00/mean/all/mAP"), ("veh AP@2", "det/dt+0.00/vehicle/all/ap@2"), ("ped AP@2", "det/dt+0.00/pedestrian/all/ap@2"),
     ("drivable", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"),
     ("lane div@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5")]
print("| run | control | " + " | ".join(k for k, _ in K) + " |")
print("|---|---|" + "---|" * len(K))
for run in ("t2_ref_s0", "t2_cam_off", "t2_radar_off", "t3_eval_r4soa"):
    for f in [f"runs/{run}/eval/metrics.json"] + sorted(glob.glob(f"runs/{run}/eval/metrics_ctl_*.json")):
        if not os.path.exists(f):
            continue
        m = json.load(open(f))["metrics"]
        ctl = os.path.basename(f)[len("metrics_ctl_"):-5] if "_ctl_" in f else "(none)"
        print(f"| {run[3:]} | {ctl} | " + " | ".join(f"{m.get(k, float('nan')):.3f}" for _, k in K) + " |")
