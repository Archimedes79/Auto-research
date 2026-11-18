#!/bin/bash
# E358 cheap VAL gate (CPU, diagnostic by hand, IN-SAMPLE on VAL, never a claim): the attached-end records of the stored dump and E325's scene-wise 5-fold cross-validation on them, both seeds.   wsl bash temp/e358/gate.sh
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
for S in 0 1; do
  nice -n 10 bash tools/env/_py14.sh jwire_data.py val --seed $S --pair-run e324_s_s$S --ends att > temp/e358/gate_val_s$S.log 2>&1
  echo "gate val seed $S exit $? $(date +%H:%M:%S)" >> temp/e358/gate_done.txt
  nice -n 10 bash tools/env/_py14.sh jwire_train.py cv --seed $S --pair-run e324_s_s$S --ends att > temp/e358/gate_cv_s$S.log 2>&1
  echo "gate cv seed $S exit $? $(date +%H:%M:%S)" >> temp/e358/gate_done.txt
done
echo "GATE DONE" >> temp/e358/gate_done.txt
