"""scratch: the written rule of the t17 card (section 1) applied to the stored tables of t16_full_s0 (a) and t17_data (b), plus the train-to-val gap of both.
    wsl python3 bev_fusion/temp/t16/rule_t17.py > bev_fusion/temp/t16/rule_t17.txt"""
import re


def load(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
        if m:
            d[m.group(1)] = float(m.group(2))
    return d


A, B = "t16_full_s0", "t17_data"
va, vb = (load(f"bev_fusion/runs/{r}/eval/by_n/table_n1000.txt") for r in (A, B))
ta, tb = (load(f"bev_fusion/runs/{r}/eval/table_train.txt") for r in (A, B))
MAPS = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", 0.010), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", 0.020),
        ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0", 0.010), ("ped. crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", 0.010),
        ("lane divider IoU@0.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", 0.020), ("road divider IoU@1", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1", 0.010)]
print("VAL 1000 frames, GT v8, one seed each; band = the card's band; 'better' = b - a > band")
better = 0
for n, k, band in MAPS:
    d = vb[k] - va[k]
    better += d > band
    print(f"  {n:24s} a {va[k]:.4f}  b {vb[k]:.4f}  b-a {d:+.4f}  band {band:.3f}  {'BETTER' if d > band else ('worse' if d < -band else 'inside band')}")
mapk = "detf/dt+0.00/mean/all/mAP"
dm = vb[mapk] - va[mapk]
print(f"  front mAP (fusion area)   a {va[mapk]:.4f}  b {vb[mapk]:.4f}  b-a {dm:+.4f}  band 0.010  {'BETTER' if dm > 0.010 else 'not better'}")
ap2 = [("vehicle AP@2", "detf/dt+0.00/vehicle/all/ap@2"), ("pedestrian AP@2", "detf/dt+0.00/pedestrian/all/ap@2")]
for n, k in ap2:
    print(f"  {n:24s} a {va[k]:.4f}  b {vb[k]:.4f}  b-a {vb[k] - va[k]:+.4f}")
n_ok = better + (1 if dm > 0.010 else 0)
print(f"\nmaps better than their band: {better} of 6; mAP better: {dm > 0.010}  -> 'at least four of the six map IoUs and front mAP2' : {better >= 4 and dm > 0.010}")
GUARDS = [("vehicle moving EPE (centre, m/s)", "vel/dt+0.00/centre/field/moving/epe", 0.3, +1), ("static EPE (cells, m/s)", "vel/dt+0.00/cells/static/epe", 0.1, +1), ("front mAP", mapk, 0.010, -1)]
worse = 0
for n, k, band, sign in GUARDS:
    d = vb[k] - va[k]
    bad = d * sign > band if sign > 0 else d < -band
    worse += bad
    print(f"  guard {n:34s} a {va[k]:.4f}  b {vb[k]:.4f}  b-a {d:+.4f}  band {band}  {'WORSE' if bad else 'holds'}")
print(f"guards worse: {worse}")
print("\nTRAIN-minus-VAL gap (train = 1000 frames of the arm's own train scenes; IoU gaps, positive = overfit)")
for r, v, t in ((A, va, ta), (B, vb, tb)):
    g = [t[k] - v[k] for _, k, _ in MAPS if k in t]
    print(f"  {r:12s} mean map-IoU gap {sum(g) / len(g):+.4f}  per map: " + " ".join(f"{x:+.3f}" for x in g))
ga = sum(ta[k] - va[k] for _, k, _ in MAPS) / 6
gb = sum(tb[k] - vb[k] for _, k, _ in MAPS) / 6
print(f"  gap change {100 * (gb - ga) / ga:+.0f} % (the card predicted at least -20 %)")
