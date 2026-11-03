import re, sys
def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
        if m: d[m.group(1)] = float(m.group(2))
    return d
run = sys.argv[1]
v = load(f"bev_fusion/runs/{run}/eval/by_n/table_n1000.txt"); t = load(f"bev_fusion/runs/{run}/eval/table_train.txt")
rows = [("drivable IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", 1), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", 1), ("carpark IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0", 1),
        ("ped crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", 1), ("lane divider IoU@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", 1), ("road divider IoU@1", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1", 1),
        ("mAP (fusion area)", "detf/dt+0.00/mean/all/mAP", 1), ("vehicle AP@2", "detf/dt+0.00/vehicle/all/ap@2", 1), ("pedestrian AP@2", "detf/dt+0.00/pedestrian/all/ap@2", 1),
        ("vel EPE moving centres", "vel/dt+0.00/centre/field/moving/epe", -1), ("vel EPE static cells", "vel/dt+0.00/cells/static/epe", -1)]
print(f"{'metric':28s} {'train':>8s} {'val':>8s} {'gap(train-val)':>15s}")
gaps = []
for n, k, s in rows:
    if k in v and k in t:
        g = (t[k] - v[k]) * s
        print(f"{n:28s} {t[k]:8.3f} {v[k]:8.3f} {g:15.3f}")
        if n.endswith(("IoU", "IoU@0.5", "IoU@1")): gaps.append(g)
print("mean gap over the six map IoUs:", round(sum(gaps) / len(gaps), 3))
