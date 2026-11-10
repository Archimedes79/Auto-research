#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion_lane_center/experiment_execution/DECISIONS.md"
python3 bev_fusion/temp/cleanup/fix_decisions.py || exit 1
F=bev_fusion/temp/cleanup/DECISIONS_fixed.md
echo "--- fixed vs e5a13211 (expect: one changed line, the D67 row) ---"
diff <(git show e5a13211:$P) $F | cut -c1-90
echo "--- fixed vs HEAD (expect: only the D68 row added) ---"
diff <(git show HEAD:$P) $F | cut -c1-90
