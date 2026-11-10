import json, os, sys
runs = sys.argv[1:] or ["t2_ref_s0", "t2_ref_s1", "t3_ref_long", "t3_eval_r4soa", "t1_r4_soa"]
keys = [("drivable", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0"), ("walkway", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0"),
        ("carpark", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0"), ("ped_cross", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0"),
        ("lane_div@.5", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5"), ("road_div@.5", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@0.5")]
print("%-15s %6s %5s %6s | %s" % ("run", "steps", "accum", "scenes", " ".join("%11s" % k for k, _ in keys)))
for r in runs:
    p = f"runs/{r}/eval/metrics.json"
    c = f"runs/{r}/config.json"
    if not os.path.exists(p):
        print(r, "no metrics"); continue
    m = json.load(open(p))["metrics"]
    cfg = json.load(open(c)) if os.path.exists(c) else {}
    ntr = cfg.get("train_scenes")
    print("%-15s %6s %5s %6s | %s   (n=%s, step %s)" % (r, cfg.get("steps"), cfg.get("accum"), len(ntr) if ntr else "all", " ".join("%11.3f" % m.get(k, float("nan")) for _, k in keys), m.get("info/n_samples"), m.get("info/step")))
