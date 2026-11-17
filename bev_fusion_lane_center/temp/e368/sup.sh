#!/bin/bash
# E368 S2 supervisor: the five runs ONE AFTER ANOTHER (one GPU job family at a time: the E362 lesson), each kept alive until its training is done (run.py resumes a failed id from the 500-step checkpoint).
# usage: wsl bash temp/e368/sup.sh      order: z_s0, r_s0, z_s1, r_s1, z2_s0
cd "$(dirname "$0")/../.."
L=temp/e368/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e368 sup] $*" >> $L; }
exec 8>/tmp/e368_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
HYP="E368 S2: the E365 token head with ONE cross-attention block after self-attention layer 1 to the ego-warped token states of the previous keyframe (the head's own states carried recursively, sequences of 3 keyframes, one input per step) decides weak tokens better than the same head trained on the same sequences without the block"
PRED="topo F1 of r minus z about +0.004 (hit window -0.006 to +0.020; recall ceiling of step 0 +3.2 points), parameters +17.4 k, +22 percent per training step (launch gate 0.268 against 0.220 s/step); odds KEEP 15, TIE 65, CUT 20 percent"
RULE="E368 card section 0 (by hand: written before the runs): KEEP = r - z >= +0.010 topo F1 on BOTH seeds at the primary read-out (each arm at its own TRAIN-selected decode, stream mode) with the paired bootstrap interval above 0, AP not below, guards held and the noise band z2 below 0.010; CUT = r - z <= -0.010 on both seeds or a guard broken on both; TIE otherwise (default: z, the cheaper)"
job() {  # name arm seed init flag
  local id=$1 name=$2 seed=$3 init=$4 flag=$5 n=0
  say "start $id"
  while ! grep -q "done $name\$" runs/$name/log.txt 2>/dev/null; do
    n=$((n+1)); [ $n -gt 6 ] && { say "$id: 6 launches, giving up"; return 1; }
    say "$id launch $n"
    $RUN --id $id --seed $seed --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jpast_train.py --name $name --seed $seed --init-seed $init $flag --seq 3 --bs 4 --steps 8000 --lr 1e-3 --ident e292_a --workers 4 >> temp/e368/train_$name.out 2>&1
    sleep 15
  done
  say "$id done"
}
job E368.z_s0 e368_z_s0 0 0 --no-past || exit 1
job E368.r_s0 e368_r_s0 0 0 "" || exit 1
job E368.z_s1 e368_z_s1 1 1 --no-past || exit 1
job E368.r_s1 e368_r_s1 1 1 "" || exit 1
job E368.z2_s0 e368_z2_s0 0 1 --no-past || exit 1
say "all five runs done"
