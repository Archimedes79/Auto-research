#!/bin/bash
# Loop 20, iteration 3: E240 seed pair 3 (and the E241 checkpoints of seed 3), launched through run.py chains.
# The two ids were cancelled in loop 19 (stalled records); run.py resumes a stalled / failed id. The loader default is now the exact distance map (D14);
# pairs 0-2 trained on the stored map, so pair 3 is pinned to it with LCR_DIST_DIR (the card of each run records env.dist_dir) to keep the pairs comparable.
cd "$(dirname "$0")/../.."
export LCR_DIST_DIR="$HOME/exp_crn_b123/feats"
RUN="python3 ../autoresearch/tools/run.py --gpu-max-mb 9000 --retries 0 --gpu"
T="bash temp/loop19/train_j1.sh"
H40="E240: the recall weight 3 gain of j1rc over the equal-budget j1eq is measured on four paired seeds with every headline metric (seed noise = measurement floor)"
P40="Q j1rc - j1eq = +0.006 (mean of 4 paired seeds), lanes found +0.06, seed SD of Q <= 0.004"
R40="Q a - c >= 0.005 and lanes found >= +0.04 with topo_F1 not lower by more than 0.01 and F1@1.0 not lower by more than 0.005 (decide.py --experiment E240)"
nohup $RUN --id E240.a3 --seed 3 --hyp "$H40" --pred "$P40" --rule "$R40" -- $T e24_D7_j1rcs3 3 3.0 e24_j1rc_s3 > /dev/null 2>&1 &
sleep 4
nohup $RUN --id E240.c3 --seed 3 --after E240.a3 --hyp "$H40" --pred "$P40" --rule "$R40" -- $T e24_D7_j1eqs3 3 1.0 e24_j1eq_s3 > /dev/null 2>&1 &
sleep 1
echo "queued: E240.a3 -> E240.c3 (stored distance map)"
