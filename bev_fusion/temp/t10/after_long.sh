#!/bin/bash
# AFTER the chained evaluation t10.dual_long_s0.eval (cold VAL 1000) has finished: the paired comparisons of card 3c item c and the decide.py line of item a (printed, not written: read it, then repeat with --write).
# CPU only, nice, 2 threads (about 6 min per compare). Run from anywhere: wsl bash temp/t10/after_long.sh
cd "$(dirname "$0")/../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
L=runs/t10_dual_long_s0
[ -f $L/eval/card.json ] || { echo "no evaluation card yet: $L/eval/card.json"; exit 1; }
echo "GT signature of the card: $($PY -c "import json;print(json.load(open('$L/eval/card.json'))['protocol']['gt_signature'])")  (must be 0.5_4_15efbde2d1)"
for ref in s0 s1; do
  nice -n 10 $PY -u evaluation/bev_suite.py compare runs/t10_dual_$ref $L --n 1000 > temp/t10/compare_long_vs_$ref.txt 2>&1
  echo "compare vs $ref exit $? -> temp/t10/compare_long_vs_$ref.txt"
done
nice -n 10 $PY -u evaluation/bev_suite.py front runs/t10_dual_s0 runs/t10_dual_s1 $L runs/t11_vel_hurdle_acc2 --n 1000 --x-max 50 --out experiments/t10_dual_features/results_x50_long.md > temp/t10/front_long.log 2>&1
echo "front exit $? -> experiments/t10_dual_features/results_x50_long.md"
v=$($PY -c "import json;m=json.load(open('$L/eval/metrics.json'));mm=m.get('metrics',m);print(mm['seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5'])")
echo "lane divider IoU@0.5 of the long run: $v (dual s0 / s1: 0.3352 / 0.3498, mean 0.3425)"
AR_PROJECT=../bev_fusion python3 ../autoresearch/tools/decide.py --variant $v --control 0.3352,0.3498 --band 0.029 --id t10.dual_long_s0
echo "(to write the verdict into the ledger: the same decide.py call with --write)"
