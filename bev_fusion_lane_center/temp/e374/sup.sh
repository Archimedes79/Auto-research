#!/bin/bash
# E374 S3 supervisor: the five runs ONE AFTER ANOTHER, each kept alive until its training is done (run.py resumes a failed id from the 500-step checkpoint). usage: wsl bash temp/e374/sup.sh   order: z_s0, m_s0, z_s1, m_s1, z2_s0
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e374/chain.log
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') [e374 sup] $*" >> $L; }
exec 8>/tmp/e374_sup.lock
flock -n 8 || { say "supervisor already runs"; exit 0; }
HYP="E374 S3: the E365 token head with ONE cross-attention block after self-attention layer 2 to road-level SD-map tokens (sd_lib carriageway centre lines, lane count, the junction connector ranges cut; trained on corrupted maps incl. absent, count-unknown and gross frames) decides weak tokens better than the same head trained by the same trainer without the block"
PRED="topo F1 of m minus z about +0.006 (hit window 0.000 to +0.012); parameters +20.3 k, +57 percent per training step (launch gate 0.155 against 0.099 s/step); odds KEEP 25, TIE 60, CUT 15 percent"
RULE="E374 card section 0 (by hand: written before the runs): KEEP = m - z >= +0.010 topo F1 on BOTH seeds at the primary read-out (each arm at its own TRAIN-selected decode, VAL error map) with the paired bootstrap interval above 0, AP not below, guards held, the noise band z2 below 0.010 and the map of ANOTHER scene (S) at least 0.005 below the record map; CUT = m - z <= -0.010 on both seeds or a guard broken on both; TIE otherwise (default z, the cheaper)"
job() {  # id name seed init flag
  local id=$1 name=$2 seed=$3 init=$4 flag=$5 n=0
  say "start $id"
  while ! grep -q "done $name\$" runs/$name/log.txt 2>/dev/null; do
    n=$((n+1)); [ $n -gt 6 ] && { say "$id: 6 launches, giving up"; return 1; }
    say "$id launch $n"
    $RUN --id $id --seed $seed --gpu --ram-mb 6000 --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_pyr.sh jmap_train.py --name $name --seed $seed --init-seed $init $flag --steps 12000 --bs 8 --lr 1e-3 --ident e292_a --workers 4 >> temp/e374/train_$name.out 2>&1
    sleep 15
  done
  say "$id done"
}
job E374.z_s0 e374_z_s0 0 0 --no-map || exit 1
job E374.m_s0 e374_m_s0 0 0 "" || exit 1
job E374.z_s1 e374_z_s1 1 1 --no-map || exit 1
job E374.m_s1 e374_m_s1 1 1 "" || exit 1
job E374.z2_s0 e374_z2_s0 0 1 --no-map || exit 1
say "all five runs done"
