#!/bin/bash
# E368 read-out, CPU part: for every arm whose dumps are done: the every-3 TRAIN dump, the E366-grid sweep (run.py ids E368.sw_<tag>), the cell. usage: wsl bash temp/e368/sweeps.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e368/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e368 sweeps] $*" >> $L; }
HYP="E368 read-out: the TRAIN-sample sweep (E366 grid, 291 frames) of the arm's own stream dump, to choose its cell by the written rule"
PRED="the cell is chosen by the rule of the E368 card; no number is predicted for the sweep"
RULE="E368 card section 0 (by hand): the cell with the highest TRAIN topo F1 among those with dup_frac <= 0.05, ties to the record cell; applied once to VAL"
for t in "z_s0 e368_z_s0 0" "r_s0 e368_r_s0 0" "z_s1 e368_z_s1 1" "r_s1 e368_r_s1 1" "z2_s0 e368_z2_s0 0"; do
  set -- $t
  tag=$1; name=$2; seed=$3
  until grep -q "$tag dumps exit 0" $L; do sleep 20; done
  [ -f runs/$name/token_train_every3.pkl ] || bash tools/env/_py14.sh temp/e368/mk_every3.py $name >> temp/e368/sweeps.out 2>&1
  say "$tag sweep"
  $RUN --id E368.sw_$tag --seed $seed --retries 0 --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py sweep --run $name --seed $seed --split train --every 3 >> temp/e368/sweeps.out 2>&1
  say "$tag sweep exit $?"
done
say "sweeps done"
