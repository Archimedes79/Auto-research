#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1 BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center
source $HOME/miniconda3/etc/profile.d/conda.sh; conda activate gcar
W=temp/_session_scratch_d113/work3
python $W/run_subset.py training/grid/test_lane_map.py "cap is 200" > $W/sub_tl.txt 2>&1
echo "exit $?" >> $W/sub_tl.txt
