#!/bin/bash
# E391 step 2 card v2 chain (D136): waits for the GPU window (22:00; no GPU start after 05:00), trains the fusion arms P / F1 / F2 / F3 on both seeds (GPU, 2 at a time),
# then the TRAIN-holdout reads of all eight (CPU, BEFORE any VAL read), the choice on the holdout (temp/e391/choose391.py, mechanical), then ONE VAL read of the chosen arm(s).
# usage: bash temp/e391/chain391.sh   (from the problem folder, in WSL)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e391/chain391.log
say() { echo "$(date '+%m-%d %H:%M:%S') [chain391] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
PYG="env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
HYP="E391 step 2 (card v2, D130 / D135 / D136): a fusion head associates the champion's delivered tokens to the segments of the frozen E390 road-graph encoding (with 'unassigned' for false tokens) better than the zero-shot E390 head, and the map read through it improves lanes per direction and the route role over perception alone; perception untouched (token inputs detached)"
PRED="(card v2.1 section 5.1 item 6, before any fusion run) assoc_acc F1 about 0.83, P about 0.81; lanes F2 / F3 about 0.92; route F2 / F3 about 0.85; PASS 15 %, PARTLY 25 %, INCONCLUSIVE 25 %, NO FUSION 35 %"
RULE="E391 card v2.1 section 5.1 (bars from the nulls measured before any fusion training, verdict by hand): clause A per chosen arm and seed at dec on VAL: assoc_acc >= 0.8427 / 0.8397 with the paired interval vs p6gate (zero-shot E390 head + no-map false-token gate) above 0 and false_unassigned <= 0.05; L lanes_acc >= 0.9182 / 0.9044; R route_f1 >= 0.8771 / 0.8750; arms chosen on the TRAIN holdout first (temp/e391/choose391.py); order NO FUSION > INCONCLUSIVE > PARTLY > PASS; VAL, no TEST"
in_window() { H=$((10#$(date +%H))); [ $H -ge 22 ] || [ $H -lt 5 ]; }
exec 8>/tmp/e391_chain.lock
flock -n 8 || { say "already runs"; exit 0; }
say "starts, waits for the GPU window (22:00)"
until in_window; do sleep 60; done
say "GPU window open"
for S in 0 1; do
  for PAIR in "P F1" "F2 F3"; do
    for A in $PAIR; do
      if [ -f runs/e391_${A}_s$S/model.pt ]; then say "${A}_s$S exists, skipped"; continue; fi
      if ! in_window; then say "past 05:00: ${A}_s$S not started"; continue; fi
      ( $RUN --id E391.${A}_s$S --seed $S --retries 0 --require-card --gpu --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYG mapassoc_probe.py train --arm $A --seed $S" >> temp/e391/${A}_s$S.out 2>&1; say "${A}_s$S exit $?" ) &
      sleep 20
    done
    wait
  done
done
for S in 0 1; do
  [ -f runs/e391_P_s$S/model.pt ] && [ -f runs/e391_F1_s$S/model.pt ] && [ -f runs/e391_F2_s$S/model.pt ] && [ -f runs/e391_F3_s$S/model.pt ] || { say "a model of seed $S is missing: stop before the reads"; exit 1; }
done
for S in 0 1; do                                     # the TRAIN-holdout reads (run.py split 'val': the read itself is on the TRAIN holdout scenes), BEFORE any VAL read
  for PAIR in "P F1" "F2 F3"; do
    for A in $PAIR; do
      ( $RUN --id E391.hold_${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY mapassoc_probe.py read --arm $A --seed $S --split holdout" >> temp/e391/hold_${A}_s$S.out 2>&1; say "hold_${A}_s$S exit $?" ) &
      sleep 15
    done
    wait
  done
done
ARMS=$(python3 temp/e391/choose391.py) || { say "the choice failed"; exit 1; }
say "choice on the holdout (temp/e391/choice391.json): VAL reads of: $ARMS"
for A in $ARMS; do
  for S in 0 1; do
    ( $RUN --id E391.read_${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 120 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY mapassoc_probe.py read --arm $A --seed $S --split val" >> temp/e391/read_${A}_s$S.out 2>&1; say "read_${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
done
say "done"
