#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2
L=temp/_session_scratch_d113
for s in 0 1; do
  bash tools/env/_py14.sh $L/step0_past.py $s val > $L/coder_ref0_$s.log 2>&1
  bash tools/env/_py14.sh $L/step0b_past.py $s > $L/coder_ref0b_$s.log 2>&1
  bash tools/env/_py14.sh lane_past_data.py potential --split val --seed $s --out temp/_session_scratch_d113/coder_potential_$s.json > $L/coder_pot_$s.log 2>&1
done
echo finished > $L/coder_t5.done
