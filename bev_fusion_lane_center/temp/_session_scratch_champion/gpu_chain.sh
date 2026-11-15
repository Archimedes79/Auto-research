#!/bin/bash
# the champion's GPU numbers: idle-GPU timing, then the 873-frame VAL eval of both seeds; diagnostics through run.py (no card, no verdict)
cd /mnt/c/Auto-research/bev_fusion_lane_center
S=/mnt/c/Auto-research/bev_fusion_lane_center/temp/_session_scratch_champion
R="python3 ../autoresearch/tools/run.py --no-card --gpu --gpu-max-mb 2000 --split val --retries 0"
$R --id lanegraph.timing_gpu --seed 0 --timeout 30 -- nice -n 19 bash tools/env/_py14.sh training/lanegraph/scripts/lg_timing.py --seed 0 --device cuda --frames 12 --rounds 8 > $S/gpu_timing.out 2>&1
for s in 0 1; do
  OMP_NUM_THREADS=2 $R --id lanegraph.eval873_s$s --seed $s --timeout 150 --result $S/eval873_s$s.json -- nice -n 19 bash tools/env/_py14.sh training/lanegraph/scripts/lg_eval.py --seed $s --frames 873 --device cuda --out $S/eval873_s$s.json > $S/eval873_s$s.out 2>&1
done
echo CHAIN_DONE
