#!/bin/bash
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
rm -f $O/e116c_data.done
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate gcar
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR="$HOME/.numba_cache_gcar"
mkdir -p $O/e116_valfail
for pn in dark blur cam_front3_off; do
  python eval_pert_val.py --pert $pn --no_graph 1 >> $O/e116_valfail/dense.log 2>&1
done
python p13_temporal_val.py --perts dark,blur,cam_front3_off --variants single,ema_hold_mon:0.5,maxdecay:0.95 --out e116_valfail --tokens > $O/e116_valfail/run.log 2>&1
echo done > $O/e116c_data.done
