R = "/mnt/c/Auto-research/bev_fusion/runs"
runs = ["t10_dual_s0", "t10_dual_s1", "t11_vel_hurdle", "t11_vel_hurdle_acc2"]
keys = ["seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0",
        "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5",
        "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@1", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1",
        "seg_obj/seg/dt+0.00/vehicle/all/all/iou@0", "seg_obj/seg/dt+0.50/vehicle/all/all/iou@0",
        "det/dt+0.00/mean/all/mAP", "det/dt+0.00/vehicle/all/ap@2", "det/dt+0.00/pedestrian/all/ap@2",
        "det/dt+0.00/vehicle/all/ate", "det/dt+0.00/vehicle/all/aoe", "det/dt+0.00/vehicle/all/ave"]
T = {}
for r in runs:
    d = {}
    for line in open(f"{R}/{r}/eval/table.txt"):
        p = line.split()
        if len(p) == 2:
            try: d[p[0]] = float(p[1])
            except ValueError: pass
    T[r] = d
print("key".ljust(52), *[r[-12:].rjust(12) for r in runs])
for k in keys:
    print(k[-52:].ljust(52), *[(f"{T[r][k]:.4f}" if k in T[r] else "-").rjust(12) for r in runs])
