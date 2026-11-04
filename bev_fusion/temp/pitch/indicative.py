"""Indicative share of the SOA map IoU: our VAL-1000 numbers (stored eval) over GaussianCaR's 400-frame numbers (t6 map_ref json). Different frame sets: indicative only."""
import json
LAY = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
       ("crossing", "ped_crossing", "iou@0"), ("lane divider @0.5", "lane_divider", "iou@0.5"), ("road divider @0.5", "road_divider", "iou@0.5")]
g = json.load(open("temp/soa_gap/map_ref_r4soa_400.json"))["systems"]["gcar"]
rows = {}
for run in ("t2_ref_s0", "t8_ctl_s0", "t8_bevfeat_s0", "t10_dual_s0"):
    m = json.load(open(f"runs/{run}/eval/metrics.json"))["metrics"]
    rows[run] = [m[f"seg_fov/seg/dt+0.00/{l}/all/all/{t}"] for _, l, t in LAY]
soa = [g[l]["all"]["0.5"][t] for _, l, t in LAY]
print("layer".ljust(20), "SOA".rjust(6), *[r.rjust(14) for r in rows])
for i, (n, _, _) in enumerate(LAY):
    print(n.ljust(20), f"{soa[i]:.3f}".rjust(6), *[f"{rows[r][i]:.3f} ({rows[r][i] / soa[i]:.2f})".rjust(14) for r in rows])
