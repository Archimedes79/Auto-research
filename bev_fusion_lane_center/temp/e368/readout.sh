#!/bin/bash
# E368 read-out chain: for every trained run the GPU dumps (TRAIN sample + VAL stream, + the checkpoint-control modes for the r arms), then the TRAIN sweep (E366 grid) on the CPU. usage: wsl bash temp/e368/readout.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e368/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e368 readout] $*" >> $L; }
HYP="E368 read-out: the trained S2 head (arm r) or its control (arm z) run frame by frame in the stream mode on the E365 cache order, dumped for the standard scorer; the cell is chosen on the arm's own TRAIN sample by the E366 rule and applied once to VAL"
PRED="topo F1 of r minus z about +0.004 (hit window -0.006 to +0.020), AP +0.01, stream within 0.005 of truncated"
RULE="E368 card section 0 (by hand: written before the runs): KEEP = r - z >= +0.010 topo F1 on both seeds at the primary read-out with the paired bootstrap interval above 0, AP not below, guards held; CUT = r - z <= -0.010 on both seeds; TIE otherwise (default z)"
one() {  # tag run seed arm
  local tag=$1 name=$2 seed=$3 arm=$4
  local extra=""
  if [ "$arm" = r ]; then
    extra="&& bash tools/env/_pyr.sh jpast_eval.py dump --run $name --seed $seed --split val --mode off && bash tools/env/_pyr.sh jpast_eval.py dump --run $name --seed $seed --split val --mode shuffled && bash tools/env/_pyr.sh jpast_eval.py dump --run $name --seed $seed --split val --mode still && bash tools/env/_pyr.sh jpast_eval.py dump --run $name --seed $seed --split val --mode truncated"
  fi
  say "$tag dumps"
  $RUN --id E368.dump_$tag --seed $seed --gpu --retries 1 --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "bash tools/env/_pyr.sh jpast_eval.py dump --run $name --seed $seed --split train --mode stream && bash tools/env/_pyr.sh jpast_eval.py dump --run $name --seed $seed --split val --mode stream $extra" >> temp/e368/readout_$tag.out 2>&1
  say "$tag dumps exit $?"
  $RUN --id E368.sweep_$tag --seed $seed --retries 0 --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py sweep --run $name --seed $seed --split train --every 3 >> temp/e368/readout_$tag.out 2>&1
  say "$tag sweep exit $?"
}
until grep -q "all five runs done" $L; do sleep 30; done
for t in "z_s0 e368_z_s0 0 z" "r_s0 e368_r_s0 0 r" "z_s1 e368_z_s1 1 z" "r_s1 e368_r_s1 1 r" "z2_s0 e368_z2_s0 0 z"; do
  set -- $t
  one $1 $2 $3 $4
done
say "read-out chain done"
