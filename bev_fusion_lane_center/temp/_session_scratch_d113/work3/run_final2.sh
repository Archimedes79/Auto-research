#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1
W=temp/_session_scratch_d113/work3
bash tools/env/_py14.sh test_jmap.py selftests > $W/final_tj.txt 2>&1
echo "exit $?" >> $W/final_tj.txt
bash tools/env/_py14.sh test_jmap_train.py selftests > $W/final_tm.txt 2>&1
echo "exit $?" >> $W/final_tm.txt
bash tools/env/_py14.sh lane_map_data.py potential --split val --seed 0 --out $W/pot_full2.json > $W/pot_out2.txt 2>&1
echo "exit $?" >> $W/pot_out2.txt
