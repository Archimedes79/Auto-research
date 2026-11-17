#!/bin/bash
# E374 S3 read-out chain: for every arm as soon as its training is done: the GPU dumps (TRAIN sample + VAL with the record map, + the map controls for the m arms), the every-3 TRAIN file, the E366-grid sweep on the CPU. usage: wsl bash temp/e374/readout.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e374/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e374 readout] $*" >> $L; }
HYP="E374 read-out: the trained S3 head (arm m: block + simulated road map) or its control (arm z) run frame by frame on the E365 cache order, dumped for the standard scorer; the cell is chosen on the arm's own TRAIN sample by the E366 rule and applied once to VAL"
PRED="topo F1 of m minus z about +0.006 (hit window 0.000 to +0.012); the map of another scene (S) at least 0.005 below the record map"
RULE="E374 card section 0 (by hand: written before the runs): KEEP = m - z >= +0.010 topo F1 on both seeds at the primary read-out with the paired bootstrap interval above 0, AP not below, guards held, noise band below 0.010 and S at least 0.005 below err; CUT = m - z <= -0.010 on both seeds; TIE otherwise (default z)"
one() {  # tag run seed arm
  local tag=$1 name=$2 seed=$3 arm=$4
  until grep -q "\[e374 sup\] E374.$tag done" $L; do sleep 30; done
  local extra=""
  if [ "$arm" = m ]; then
    for m in clean gross none S U C2; do extra="$extra && bash tools/env/_pyr.sh jmap_eval.py dump --run $name --seed $seed --split val --mode $m"; done
  fi
  say "$tag dumps"
  $RUN --id E374.dump_$tag --seed $seed --gpu --retries 1 --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "bash tools/env/_pyr.sh jmap_eval.py dump --run $name --seed $seed --split train --mode err && bash tools/env/_pyr.sh jmap_eval.py dump --run $name --seed $seed --split val --mode err $extra" >> temp/e374/readout_$tag.out 2>&1
  say "$tag dumps exit $?"
  bash tools/env/_py14.sh temp/e368/mk_every3.py $name >> temp/e374/readout_$tag.out 2>&1
  $RUN --id E374.sw_$tag --seed $seed --retries 0 --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh jtoken_eval.py sweep --run $name --seed $seed --split train --every 3 >> temp/e374/readout_$tag.out 2>&1
  say "$tag sweep exit $?"
}
for t in "z_s0 e374_z_s0 0 z" "m_s0 e374_m_s0 0 m" "z_s1 e374_z_s1 1 z" "m_s1 e374_m_s1 1 m" "z2_s0 e374_z2_s0 0 z"; do
  set -- $t
  one $1 $2 $3 $4
done
say "read-out chain done"
