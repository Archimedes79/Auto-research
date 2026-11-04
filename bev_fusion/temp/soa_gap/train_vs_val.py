import json
import sys

run = sys.argv[1]
va = json.load(open(f"{run}/eval/metrics.json"))["metrics"]
tr = json.load(open(f"{run}/eval/metrics_train.json"))["metrics"]
print(f"{run}: TRAIN subset (n={tr.get('info/n_samples')}, scenes {tr.get('info/n_scenes')}) vs VAL (n={va.get('info/n_samples')}, scenes {va.get('info/n_scenes')}), dt 0, camera cells")
for layer, tol in (("drivable_area", "iou@0"), ("walkway", "iou@0"), ("carpark_area", "iou@0"), ("ped_crossing", "iou@0"),
                   ("lane_divider", "iou@0.5"), ("road_divider", "iou@0.5")):
    k = f"seg_fov/seg/dt+0.00/{layer}/all/all/{tol}"
    print("%-14s %-8s train %.3f   val %.3f" % (layer, tol, tr[k], va[k]))
for k in ("det/dt+0.00/mean/all/mAP", "det/dt+0.00/vehicle/all/ap@2", "det/dt+0.00/pedestrian/all/ap@2"):
    print("%-40s train %.3f   val %.3f" % (k, tr[k], va[k]))
for k in sorted(va):
    if k.startswith("val_loss/group."):
        print("%-40s train %.3f   val %.3f" % (k, tr.get(k, float("nan")), va[k]))
