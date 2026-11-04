import json
import os

for f in ("runs/speed_a3/config.json", "runs/t10_dual_s0/config.json", "runs/t16_full_s0/config.json", "runs/t20_base_s0/config.json"):
    p = os.path.join("/mnt/c/Auto-research/bev_fusion", f)
    try:
        c = json.load(open(p))
        print(f, {k: c.get(k) for k in ("seq_len", "seq_bptt", "accum", "batch_size", "workers", "steps")}, "motion_block", c.get("model", {}).get("motion_block"))
    except Exception as e:
        print(f, "-", type(e).__name__)
