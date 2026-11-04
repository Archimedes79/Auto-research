"""scratch: side-by-side of the two small runs from their stored eval tables (by_n/table_n80.txt)"""
import re, sys
def load(run):
    d = {}
    for line in open(f"bev_fusion/runs/{run}/eval/by_n/table_n80.txt", encoding="utf-8"):
        m = re.match(r"^(\S+)\s+(-?[0-9.]+|nan)\s*$", line.rstrip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d
A, B = load(sys.argv[1]), load(sys.argv[2])
keys = [k for k in A if k in B]
want = ["seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1",
        "curve/dt+0.00/walkway/roc_auc", "curve/dt+0.00/ped_crossing/roc_auc", "curve/dt+0.00/carpark_area/roc_auc", "curve/dt+0.00/drivable_area/ap",
        "curve/dt+0.00/lane_divider/ap", "curve/dt+0.00/walkway/ap",
        "detf/dt+0.00/mean/all/mAP", "detfp/dt+0.00/mean/all/mAP", "detf/dt+0.00/vehicle/all/ap@2", "detf/dt+0.00/pedestrian/all/ap@2",
        "vel/dt+0.00/centre/field/moving/epe", "vel/dt+0.00/centre/objhead/moving/epe", "vel/dt+0.00/centre/zero/moving/epe", "vel/dt+0.00/cells/static/epe",
        "vel/dt+0.00/cells/vehicle_moving/epe", "val_loss/group.areas", "val_loss/group.lines", "val_loss/group.objects", "val_loss/group.motion", "val_loss/total"]
print(f"{sys.argv[1]:>40s} {sys.argv[2]:>26s}")
for k in want:
    if k in A and k in B:
        print(f"{k:62s} {A[k]:9.4f} {B[k]:9.4f}")
    else:
        print(f"{k:62s} {'-' if k not in A else A[k]:>9} {'-' if k not in B else B[k]:>9}")
extra = [k for k in keys if k.startswith("seg_obj/seg/dt+0.00/vehicle")][:4]
for k in extra:
    print(f"{k:62s} {A[k]:9.4f} {B[k]:9.4f}")
