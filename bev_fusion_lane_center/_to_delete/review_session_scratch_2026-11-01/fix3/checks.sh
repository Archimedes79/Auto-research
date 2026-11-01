cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad
nice python3 ../autoresearch/tools/selftest.py > /tmp/ar_st_dir3.txt 2>&1; echo "autoresearch selftest exit $?"
grep -B12 -A3 'Traceback\|FAIL' /tmp/ar_st_dir3.txt | head -30
echo "--- lane decisions"
python3 $S/recheck_decisions.py
echo "--- lane card / experiment checks"
python3 ../autoresearch/tools/card.py check --all 2>&1 | grep -v "^ok" | tail -5
python3 ../autoresearch/tools/experiment.py check --all 2>&1 | grep -A3 "NOT READY"
echo "--- bev fallback"
cd ../bev_fusion
python3 -c "
import sys; sys.path.insert(0, '../autoresearch/tools')
import decide as D
for k in ['det/dt+0.00/vehicle/all/ate', 'detr/dt+0.50/pedestrian/all/ave_slow_moving', 'seg_fov/seg/dt+0.00/lane_divider:zc/all/all/field_mae', 'val_loss/total', 'ms/median_ms', 'det/dt+0.00/vehicle/all/ap@2', 'seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0', 'det/dt+0.00/vehicle/all/ave_skill_moving']:
    print(k, D.named_lower(k, '.'))
"
echo "--- bev selftest"
nice /home/ml/miniconda3/envs/gcar/bin/python evaluation/bev_selftest.py 2>&1 | tail -15
