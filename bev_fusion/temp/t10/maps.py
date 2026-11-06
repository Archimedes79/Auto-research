import json
keys = ["seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0.5", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0",
        "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5",
        "seg_obj/seg/dt+0.00/vehicle/all/all/iou@0", "seg_obj/seg/dt+0.00/pedestrian/all/all/iou@0", "det/dt+0.00/mean/all/mAP"]
runs = ["t8_ctl_s0", "t8_ctl_s1", "t10_dual_s0", "t10_dual_s1", "t8_bevfeat_s0", "t8_bevfeat_s1"]
ms = {r: json.load(open(f"runs/{r}/eval/metrics.json")) for r in runs}
def get(m, k):
    if k in m: return m[k]
    for v in m.values():
        if isinstance(v, dict) and k in v: return v[k]
print("| metric | " + " | ".join(runs) + " |")
for k in keys:
    print("| " + k.split("/")[3 if "seg" in k else 1] + " | " + " | ".join(f"{get(ms[r], k):.3f}" if get(ms[r], k) is not None else "-" for r in runs) + " |")
