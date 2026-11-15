#!/bin/bash
# scratch: gate C0 read-out: paired scene bootstrap, CPU only, one thread per lane, two detached lanes (the lane chat's training is running: side jobs stay small).
#   lane 1: stand-in 'all' against stand-in 'current' (the rule C0 comparison), then B with 'all' against B
#   lane 2: stand-in 'all' against the control t21_pair4_s0, then B with 'all' against the control
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
L=temp/t22
C="--curves --fusion --velocity --n 1000"
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_standin_pair4 runs/t22_standin_all_pair4 --all > $L/gate_rows_C0_standin.log 2>&1
$PY $L/chain/compare_gate.py runs/t21_pair4_s0 runs/t22_B_s0 runs/t22_B_all --all > $L/gate_rows_C0_B.log 2>&1
cat > $L/lane1_C0.sh <<'EOF'
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
C="--curves --fusion --velocity --n 1000"
L=temp/t22
$PY evaluation/bev_suite.py compare runs/t22_standin_pair4 runs/t22_standin_all_pair4 $C > $L/compare_C0_all_vs_current.log 2>&1
echo "lane1 a exit $?" > $L/compare_C0_lane1.done
$PY evaluation/bev_suite.py compare runs/t22_B_s0 runs/t22_B_all $C > $L/compare_C0_Ball_vs_B.log 2>&1
echo "lane1 b exit $?" >> $L/compare_C0_lane1.done
EOF
cat > $L/lane2_C0.sh <<'EOF'
cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY=$HOME/miniconda3/envs/gcar/bin/python
C="--curves --fusion --velocity --n 1000"
L=temp/t22
$PY evaluation/bev_suite.py compare runs/t21_pair4_s0 runs/t22_standin_all_pair4 $C > $L/compare_C0_all_vs_control.log 2>&1
echo "lane2 a exit $?" > $L/compare_C0_lane2.done
$PY evaluation/bev_suite.py compare runs/t21_pair4_s0 runs/t22_B_all $C > $L/compare_C0_Ball_vs_control.log 2>&1
echo "lane2 b exit $?" >> $L/compare_C0_lane2.done
EOF
rm -f $L/compare_C0_lane*.done
setsid nohup bash $L/lane1_C0.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash $L/lane2_C0.sh > /dev/null 2>&1 < /dev/null &
sleep 3
ps -eo pid,etime,args | grep -E 'bev_suite' | grep -v grep | cut -c1-170
echo "--- gate rows stand-in all (ref control, twin = stand-in current)"
sed -n '1,40p' $L/gate_rows_C0_standin.log | cut -c1-200
