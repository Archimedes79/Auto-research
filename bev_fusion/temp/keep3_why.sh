#!/bin/bash
# scratch: why are the new-line networks no keep3 candidates? gt signature and the ranking keys of their cards
cd /mnt/c/Auto-research/bev_fusion || exit 1
python3 - <<'EOF'
import json, os
k = json.load(open("records/keep3.json"))
keys = k["metric_keys"]
print("keep3 gt_signature:", k["gt_signature"], "| split", k["split"])
for r in ("t10_dual_s0", "t16_full_s0", "t20_base_s0", "t23_C_s0", "t23_C_s1", "t21_ctl_s1", "t22_B_s0", "t22_A1_s0"):
    d = os.path.join("runs", r)
    try:
        m = json.load(open(os.path.join(d, "eval", "metrics.json")))
    except Exception as e:
        print(f"{r}: no metrics.json ({type(e).__name__})"); continue
    gt = ""
    try:
        gt = open(os.path.join(d, "gt.json")).read().strip()[:70]
    except Exception:
        pass
    have = [x for x in keys if x in m]
    print(f"{r}: gt.json {gt!r} | ranking keys present {len(have)}/{len(keys)} | top-level keys of metrics.json: {sorted(set(k.split('/')[0] for k in m))[:8]} | split {m.get('split')!r}")
    for fld in ("gt_signature", "split", "frames", "subset"):
        if fld in m: print("    ", fld, "=", str(m[fld])[:80])
EOF
