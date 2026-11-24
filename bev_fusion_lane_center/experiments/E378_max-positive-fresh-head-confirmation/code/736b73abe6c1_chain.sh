#!/bin/bash
# E357 (T0 of the token design, D105): the whole night chain, every job through run.py. Card: experiments/E357_oracle-nodes-one-pair-head/experiment.md.
# cache (CPU) -> kill check at 2.0 m -> controls on VAL (and with 17 % of the nodes removed) -> wait until E354's trainings are through
# -> launch gate (50 steps) -> the two trainings (two GPU tickets) -> dump + score (thr 0.5 / 0.3 / 0.7, and the degraded read-out).
# The launch-gate rule of the card is applied here as written: above 2.0 s per step or above 9 GB reserved, both seeds run with --bs 4 --steps 6000;
# the numbers go to temp/e357/gate_result.txt (the lane chat writes them into the card at its next check).
# Never edit this script while it runs. A failed stage stops the chain (the log says where).
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e357/chain.log
PY="bash tools/env/_py14.sh"
RUN="python3 ../autoresearch/tools/run.py"
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $L; }
HYP="Given the ground-truth nodes of the 2.5 m lattice and no types, ONE pair head recovers at least half of the gap between the best link rule and the GT links on topo F1 and joins the junction links with an F1 of at least 0.75 (stage T0 of the token design E355)."
PRED="best rule topo F1 0.82 (seen on 31 frames before the run: 0.69), hi 0.955, p 0.92 (gap share about 0.75); junction-link F1 of p 0.80; the degraded read-out loses about 0.25 topo F1"
RULE="E357 card section 0 (by hand: oracle-node replay), section-free read-out, p >= 0.5: FAIL first (both seeds gap share on topo F1 < 0.5, or junction-link F1 < 0.50), then PASS (both seeds gap share >= 0.5 AND junction-link F1 >= 0.75), else INCONCLUSIVE; VAL 873, no TEST"
card_job() { id=$1; shift; $RUN --id $id --seed ${SEED:-0} --retries 0 --require-card --split val --timeout 240 --hyp "$HYP" --pred "$PRED" --rule "$RULE" "$@"; }
exec 9>/tmp/e357_chain.lock
flock -n 9 || { say "chain already runs"; exit 0; }
say "chain starts"

# 1. the cache of the node tables (derived data, CPU, 2 threads, niced: E354 trains and is host-bound)
for SP in train val; do
  $RUN --id E357.cache_$SP --retries 0 --no-card --split val --timeout 120 -- nice -n 19 $PY lane_tok_data.py build-cache --split $SP --lattice 2.5 --threads 2 >> temp/e357/chain_out.log 2>&1
  rc=$?; say "cache $SP exit $rc"; [ $rc -ne 0 ] && exit 1
done

# 2. the kill check: hi at 2.0 m against the stored card of E346's arm l2
$RUN --id E357.check --retries 0 --no-card --split val --timeout 60 -- nice -n 19 $PY jtok_eval.py ctl --lattice 2.0 --check >> temp/e357/chain_out.log 2>&1
rc=$?; say "check 2.0 m exit $rc (0 = PASS)"; [ $rc -ne 0 ] && { say "KILL: the reproduction check failed, nothing else runs"; exit 1; }

# 3. the controls on VAL: the three rules and hi, both read-outs; then the same with 17 % of the nodes removed
card_job E357.ctl -- nice -n 19 $PY jtok_eval.py ctl --lattice 2.5 >> temp/e357/chain_out.log 2>&1
rc=$?; say "ctl exit $rc"; [ $rc -ne 0 ] && exit 1
card_job E357.ctl_drop -- nice -n 19 $PY jtok_eval.py ctl --lattice 2.5 --drop 0.17 --drop-seed 0 >> temp/e357/chain_out.log 2>&1
say "ctl drop exit $?"

# 4. the launch gate, after E354's trainings (the gate must not run next to another host-bound training)
while ps -eo cmd | grep -E "python jrecur_train.py" | grep -v grep | grep -q .; do sleep 60; done
say "E354 trainings are through, the gate starts; load $(cut -d' ' -f1-3 /proc/loadavg), GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
[ -e runs/e357_gate ] && { say "runs/e357_gate exists"; exit 1; }
$RUN --id E357.gate --seed 0 --gpu --retries 0 --no-card --split val --timeout 30 -- $PY jtok_train.py --name e357_gate --seed 0 --lattice 2.5 --steps 50 --bs 8 --lr 1e-3 --workers 4 >> temp/e357/chain_out.log 2>&1
rc=$?; CL=$(grep -h "clock:" runs/e357_gate/log.txt | tail -n 1)
say "gate exit $rc: $CL"; [ $rc -ne 0 ] && exit 1
SPS=$(echo "$CL" | sed -n 's/.*clock: \([0-9.]*\) s\/step.*/\1/p'); RES=$(echo "$CL" | sed -n 's/.*reserved \([0-9.]*\) GB.*/\1/p')
[ -z "$SPS" ] && { say "the clock line could not be read"; exit 1; }
BS=8; STEPS=3000
if awk -v s="$SPS" -v r="${RES:-0}" 'BEGIN{exit !(s > 2.0 || r > 9.0)}'; then BS=4; STEPS=6000; fi
echo "gate $(date '+%Y-%m-%d %H:%M'): $SPS s/step, reserved ${RES:-?} GB -> --bs $BS --steps $STEPS; line: $CL" > temp/e357/gate_result.txt
say "gate rule: --bs $BS --steps $STEPS"

# 5. the two trainings, two GPU tickets
for S in 0 1; do
  [ -e runs/e357_p_s$S ] && { say "runs/e357_p_s$S exists"; exit 1; }
  ( SEED=$S card_job E357.p_s$S --gpu --ram-mb 6000 -- $PY jtok_train.py --name e357_p_s$S --seed $S --lattice 2.5 --steps $STEPS --bs $BS --lr 1e-3 --workers 4 >> temp/e357/train_s$S.out 2>&1; echo $? > temp/e357/train_s$S.rc ) &
  sleep 5
done
wait
say "trainings exit $(cat temp/e357/train_s0.rc) / $(cat temp/e357/train_s1.rc)"
[ "$(cat temp/e357/train_s0.rc)" = 0 ] && [ "$(cat temp/e357/train_s1.rc)" = 0 ] || exit 1

# 6. dump + score per seed (the dump waits for GPU room itself), then the degraded read-out
for S in 0 1; do
  SEED=$S card_job E357.read_s$S -- bash -c "$PY jtok_eval.py dump --run e357_p_s$S --seed $S && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.5 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.3 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.7 && $PY jtok_eval.py dump --run e357_p_s$S --seed $S --drop 0.17 --drop-seed 0 && $PY jtok_eval.py score --run e357_p_s$S --seed $S --thr 0.5 --drop 0.17 --drop-seed 0" >> temp/e357/chain_out.log 2>&1
  say "read-out seed $S exit $?"
done
say "chain done"
