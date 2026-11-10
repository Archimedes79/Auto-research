#!/bin/bash
# scratch: D: layout and free space, the per-seed values of the best arm, sizes of what a backup of the network needs (read-only)
echo "=== D: free"; df -h /mnt/d 2>/dev/null | tail -1
echo "=== D:\\Auto-research top level"; ls /mnt/d/Auto-research 2>/dev/null | head -20
echo "=== sizes of t21_ctl seeds"; du -sh /mnt/c/Auto-research/bev_fusion/runs/t21_ctl_s0 /mnt/c/Auto-research/bev_fusion/runs/t21_ctl_s1 2>/dev/null
ls /mnt/c/Auto-research/bev_fusion/runs/t21_ctl_s0 /mnt/c/Auto-research/bev_fusion/runs/t21_ctl_s1 | head -30
cd /mnt/c/Auto-research/bev_fusion || exit 1
python3 - <<'EOF'
import json
k = json.load(open("records/keep3.json"))["metric_keys"]
for r in ("t21_ctl_s0", "t21_ctl_s1", "t21_pair4_s0", "t21_pair_s0", "t23_C_s0", "t23_C_s1", "t20_base_s0"):
    m = json.load(open(f"runs/{r}/eval/metrics.json", encoding="utf-8-sig"))["metrics"]
    print(r, round(sum(m[x] for x in k) / len(k), 4))
EOF
