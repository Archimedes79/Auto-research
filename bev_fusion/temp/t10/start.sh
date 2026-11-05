#!/usr/bin/env bash
# start t10_dual_s0 under run.py (D23); waits for a GPU slot and for the GPU memory in use to fall below 3000 MB (t7_rgb_s1 is training)
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id t10.dual_s0 --gpu --gpu-max-mb 3000 --split train \
  --hyp "the frozen DualBEV BEV and the frozen BEVFusion BEV together in one trunk keep the maps of t8_bevfeat AND the objects of t8_ctl" \
  --pred "cold VAL 1000: lane divider@0.5 0.34, drivable 0.80, walkway 0.33, front mAP2 0.33 (0.25-0.38), ped AP@2 0.15" \
  --rule "E1 both (lane div, drivable >= bevfeat-0.02, walkway >= bevfeat-0.054, mAP2 >= ctl-0.054) -> seed twin; E2 maps kept, mAP2 0.22-0.31 -> branch-specific object head; E3 objects back, maps <= ctl+0.05 -> trunk ignores branch 2; E4 both mid -> 6000 steps (experiments/t10_dual_features/experiment.md)" \
  -- bash temp/train/run_ar.sh training/configs/t10_dual_s0.json > runs/t10_dual_s0.launcher.log 2>&1
echo "launcher exit $?"
