S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
bash $S/fix3/head_test.sh 2>&1 | grep -v '^[<>-]\|^[0-9]'
cd /mnt/c/Auto-research/bev_fusion_lane_center
echo "--- lane decisions"
python3 $S/recheck_decisions.py
cd ../bev_fusion
echo "--- bev selftest"
nice /home/ml/miniconda3/envs/gcar/bin/python evaluation/bev_selftest.py 2>&1 | grep -v "^drivable\|^walkway\|^lane_div\|^vehicle\|^pedestrian\|^paired" | tail -14
