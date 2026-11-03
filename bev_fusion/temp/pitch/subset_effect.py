"""Frame-subset effect of the GaussianCaR comparison: t1_r4_soa on the 400 GaussianCaR frames (map_ref json) against the same model on 1000 cold VAL frames (stored eval)."""
import json
LAY = [("drivable", "drivable_area", "iou@0"), ("walkway", "walkway", "iou@0"), ("carpark", "carpark_area", "iou@0"),
       ("crossing", "ped_crossing", "iou@0"), ("lane divider", "lane_divider", "iou@0.5"), ("road divider", "road_divider", "iou@0.5")]
j = json.load(open("temp/soa_gap/map_ref_r4soa_400.json"))["systems"]
gc = {n: j["gcar"][l]["all"]["0.5"][t] for n, l, t in LAY}
o400 = {n: j["ours"][l]["all"]["0.5"][t] for n, l, t in LAY}
import os
for run in ("t3_eval_r4soa", "t1_r4_soa"):
    p = f"runs/{run}/eval/metrics.json"
    if os.path.exists(p):
        m = json.load(open(p))["metrics"]
        n = m.get("info/n_samples")
        o1000 = {nm: m[f"seg_fov/seg/dt+0.00/{l}/all/all/{t}"] for nm, l, t in LAY}
        print(run, "frames", n)
        break
ratio = {n: o400[n] / o1000[n] for n in o400}
print("layer".ljust(14), "r4soa@400".rjust(10), f"r4soa@{n}".rjust(10), "400/n".rjust(7))
for nm in o400:
    print(nm.ljust(14), f"{o400[nm]:.3f}".rjust(10), f"{o1000[nm]:.3f}".rjust(10), f"{ratio[nm]:.2f}".rjust(7))
print()
print("dual seeds: raw share of GaussianCaR (1000 frames vs 400) | share after scaling by the subset ratio of t1_r4_soa (an estimate)")
for run in ("t10_dual_s0", "t10_dual_s1"):
    m = json.load(open(f"runs/{run}/eval/metrics.json"))["metrics"]
    raw = {nm: m[f"seg_fov/seg/dt+0.00/{l}/all/all/{t}"] for nm, l, t in LAY}
    print(run, " ".join(f"{nm.split()[0]}:{raw[nm] / gc[nm]:.2f}/{raw[nm] * ratio[nm] / gc[nm]:.2f}" for nm in raw))
print("GaussianCaR on the 400 frames:", {k: round(v, 3) for k, v in gc.items()})
