"""print the t12 numbers of the card from the three metrics.json files (check of the card's table, scratch)"""
import json

KEYS = ["epe_moving", "epe_moving_vehicle", "epe_static", "epe_centre_moving_field", "epe_centre_moving_objhead",
        "epe_centre_moving_zero", "total_main", "sem.obj_iou.vehicle@-0.5", "sem.obj_iou.vehicle@+0.0",
        "sem.obj_iou.vehicle@+0.5", "sem.obj_bce.vehicle@-0.5", "sem.obj_bce.vehicle@+0.5", "n_moving", "n_centre_moving_field"]
for name in ("t12_mb_s0", "t12_mb_s1", "t12_hurdle_ctl"):
    m = json.load(open(f"runs/{name}/metrics.json"))
    mt = m["meta"]
    print(name, "base", mt.get("base"), mt.get("ckpt"), "seed", mt.get("seed"), "steps", mt.get("steps"), "val", mt.get("val_samples"), mt.get("gt_signature"))
    for arm in [a for a in m if a != "meta"]:
        d = m[arm]
        print("  ", arm, {k: (round(d[k], 4) if isinstance(d.get(k), float) and d[k] == d[k] else d.get(k)) for k in KEYS})
    print("   meta", {k: mt.get(k) for k in ("block_ms_per_frame", "peak_gpu_gb", "gpu_used_mb_at_timing")})
