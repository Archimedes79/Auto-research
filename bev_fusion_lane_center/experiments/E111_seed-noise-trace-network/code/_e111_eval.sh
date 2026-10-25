#!/bin/bash
# Loop 12 / E111: evaluate the extra seed(s) of the trace network as soon as the TRAINING PROCESS has finished (the training saves model.pt every 2000 iterations, so the file alone is no sign of completion)
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
cd /mnt/c/ML-Experiment/bev_fusion_lane_center/training
for n in "$@"; do
  while pgrep -f "[p]10_train.py --name $n" > /dev/null; do sleep 30; done
  grep -q "^it 10000 " $O/$n/stdout.txt || { echo "training of $n did not reach 10000 iterations" > $O/$n/INCOMPLETE; continue; }
  bash _pyr.sh p10_infer.py --name $n --split val_clean > $O/$n/infer_val.log 2>&1
  bash _pyr.sh p10_infer.py --name $n --split test_clean > $O/$n/infer_test.log 2>&1
  bash _pyr.sh p10_final_eval.py --name $n > $O/$n/final_eval.log 2>&1
done
echo done > $O/e111_eval.done
