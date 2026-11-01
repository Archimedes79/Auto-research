"""One-off: derived numbers quoted in the t3 report (differences between two stored metric files and mAP2 of stored files)."""
import json
L = lambda f: json.load(open(f))["metrics"]
def m2(m):
    return sum(m[f"det/dt+0.00/{c}/all/ap@{t}"] for c in ("vehicle", "pedestrian") for t in (0.5, 1, 2, 4)) / 8
tr, va = L("runs/t1_r4_soa/eval/metrics_train.json"), L("runs/t1_r4_soa/eval/metrics.json")
print(f"t1_r4_soa mAP2: train subset ({int(tr['info/n_samples'])} samples) {m2(tr):.3f}, full val ({int(va['info/n_samples'])} frames) {m2(va):.3f}")
r1, lp = L("runs/t2_ref_s1/eval/metrics.json"), L("runs/t2_lift_p/eval/metrics.json")
k = "det/dt+0.00/pedestrian/all/ap@2"
print(f"lift_p pedestrian AP@2 {lp[k]:.3f} vs ref_s1 {r1[k]:.3f}: {lp[k] - r1[k]:+.3f}")
ro, zs, ad = L("runs/t2_radar_off/eval/metrics.json"), L("runs/t3_cam_zsum/eval/metrics.json"), L("runs/t3_cam_adapter/eval/metrics.json")
for name, key in (("walkway", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"), ("drivable", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"),
                  ("divider", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5")):
    print(f"{name}: zsum - radar_off {zs[key] - ro[key]:+.3f}; adapter - radar_off {ad[key] - ro[key]:+.3f}")
