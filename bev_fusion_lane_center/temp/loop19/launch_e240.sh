#!/bin/bash
# E240 seed-noise of the recall fine-tune: four new fine-tunes (j1rc seeds 2 / 3 at recall weight 3, j1eq seeds 2 / 3 at weight 1), one GPU training at a time
# (a second training only slows both), each followed by its VAL 873 evaluation; then the four stored seeds are re-evaluated with the same entry point.
#   bash temp/loop19/launch_e240.sh        (queues the chain with run.py --after; returns at once after queuing, runs in the background)
cd "$(dirname "$0")/../.."
RUN="python3 ../autoresearch/tools/run.py --gpu-max-mb 9000 --retries 0"
HYP="E240: the recall weight 3 gain of j1rc over the equal-budget j1eq is measured on four paired seeds with every headline metric (seed noise = measurement floor)"
PRED="Q j1rc - j1eq = +0.006 (mean of 4 paired seeds), lanes found +0.06, seed SD of Q <= 0.004"
RULE="Q a - c >= 0.005 and lanes found >= +0.04 with topo_F1 not lower by more than 0.01 and F1@1.0 not lower by more than 0.005 (decide.py --experiment E240)"
nohup $RUN --id E240.a2 --seed 2 --after E243.timing --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash temp/loop19/train_j1.sh e24_D7_j1rcs2 2 3.0 e24_j1rc_s2 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.c2 --seed 2 --after E240.a2 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash temp/loop19/train_j1.sh e24_D7_j1eqs2 2 1.0 e24_j1eq_s2 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.a3 --seed 3 --after E240.c2 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash temp/loop19/train_j1.sh e24_D7_j1rcs3 3 3.0 e24_j1rc_s3 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.c3 --seed 3 --after E240.a3 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash temp/loop19/train_j1.sh e24_D7_j1eqs3 3 1.0 e24_j1eq_s3 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.ev0 --after E240.c3 --hyp "$HYP" --pred "reproduce the stored seeds 0 / 1: evaluate.py cards of e24_j1rc_s0/s1 and e24_j1eq_s0/s1 match the E219 numbers (Q 0.739 / 0.743 and 0.735 / 0.731)" --rule "layer 1: stored numbers reproduced within 0.002; then decide.py --experiment E240" -- bash -c 'for n in e24_j1rc_s0 e24_j1rc_s1 e24_j1eq_s0 e24_j1eq_s1; do bash tools/env/_py14.sh evaluate.py $n 2>&1 | tail -n 12; done' > /dev/null 2>&1 &
sleep 1
echo "E240 chain queued: E240.a2 -> c2 -> a3 -> c3 -> ev0"
