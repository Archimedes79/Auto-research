#!/bin/bash
# smoke test of the refinement switches on D3 (20 iterations each, no evaluation)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate gcar
source /mnt/c/ML-Experiment/bev_fusion_lane_center/tools/env/bflc_env.sh
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for flags in "stage_b=true" "stage_c=true" "spf2=true" "reach=12" "b=3 mcl=true"; do
  echo "=== D3 $flags"
  python p16_train.py --design D3 --name smoke_flags --smoke --eval none --workers 2 --set $flags --w vc=1.0 2>&1 | tail -3 | cut -c1-400
done
echo "=== D1 reach (fixed) stage_b"
python p16_train.py --design D1 --name smoke_flags --smoke --eval none --workers 2 --set stage_b=true stage_c=true 2>&1 | tail -2 | cut -c1-300
