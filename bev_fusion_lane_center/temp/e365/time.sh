#!/bin/bash
# E365 G-cost timing (quiet GPU) and the parity check, one run.py job per seed 0 only for the timing; usage: bash temp/e365/time.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e365/texts.sh
PY="bash tools/env/_py14.sh"
CMD="$PY jtoken_eval.py timing --run e365_s1_s0 --b-run e292_a_s0 && $PY jtoken_eval.py check-parity --run e365_s1_s0 --seed 0"
python3 ../autoresearch/tools/run.py --id E365.time_s0 --seed 0 --gpu --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$CMD" >> temp/e365/time_s0.out 2>&1
echo "time s0 exit $?" >> temp/e365/chain.log
