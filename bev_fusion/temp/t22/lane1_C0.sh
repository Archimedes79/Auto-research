cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
C="--curves --fusion --velocity --n 1000"
L=temp/t22
$PY evaluation/bev_suite.py compare runs/t22_standin_pair4 runs/t22_standin_all_pair4 $C > $L/compare_C0_all_vs_current.log 2>&1
echo "lane1 a exit $?" > $L/compare_C0_lane1.done
$PY evaluation/bev_suite.py compare runs/t22_B_s0 runs/t22_B_all $C > $L/compare_C0_Ball_vs_B.log 2>&1
echo "lane1 b exit $?" >> $L/compare_C0_lane1.done
