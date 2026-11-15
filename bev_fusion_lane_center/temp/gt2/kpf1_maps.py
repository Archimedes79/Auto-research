import json, sys
for r in sys.argv[1:]:
    try:
        d = json.load(open(f"../../runs/{r}/jmaps_val.json"))
    except Exception as ex:
        print(r, "missing", ex)
        continue
    kp = d["keypoints"]
    f1 = (kp["kp_ours_tol3.0_noang_ent"]["f1"] + kp["kp_ours_tol3.0_noang_ext"]["f1"]) / 2
    print(f"{r:18s} kp_f1 {f1:.3f}; zone IoU {d['zone_iou']:.3f}; core IoU {d['core_iou']:.3f}; best-threshold F1 {d['kp_f1_best_threshold']['ours']:.3f}")
