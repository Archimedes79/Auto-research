#!/bin/bash
# Re-order the GPU queue of the night (21:55): after E240.c2 (seed pair 2) come E237 (exact distance map, seeds 0 / 1) and E244 arm b (clamp 8 m), then
# the E240 seed pair 3 and the stored-seed re-evaluation. Only the not yet started launchers of the first chain are replaced (they wait for --after).
cd "$(dirname "$0")/../.."
for id in E240.a3 E240.c3 E240.ev0; do
  pid=$(ps -eo pid,args | grep -E "[r]un\.py.*--id $id( |$)" | awk '{print $1}')
  [ -n "$pid" ] && { kill $pid && echo "stopped waiting launcher $id (pid $pid)"; }
done
sleep 2
RUN="python3 ../autoresearch/tools/run.py --gpu-max-mb 9000 --retries 0"
T="bash temp/loop19/train_j1.sh"
DIST="$HOME/exp_gcar_b123x/feats"
H37="E237: the j1 equal-budget fine-tune (4 u, recall weight 1) trained on the CPU-exact distance map (D10 a) gives a different Q than on the stored TF32 map"
P37="Q(exact) - Q(stored) = +0.002 (mean of 2 seeds), F1@1.0 +0.003; step 0 exposure: 8.7 % of the supervised pixels damaged (E237 step 0)"
R37="KEEP (D10 option b: exact targets for all training) if Q a - c >= +0.010 (tool band) with both seeds on the same side and F1@1.0 not lower by more than 0.005; CUT if Q a - c <= -0.005 (look for a loader error first); else TIE (keep the stored map for the history, D10 option c)"
H44="E244 arm b: lifting the clamp of the soft minimum of the recall term from 3 m to 8 m (recall weight 1) removes the gradient dead zone and lifts lanes found of the plain fine-tune"
P44="lanes found between j1eq (0.674) and j1rc (0.736), Q within +-0.005 of j1eq"
R44="if lanes found of b >= j1eq + 0.04 the dead zone is part of the lever (build the phd loss); if b equals j1eq the clamp is not the lever; Q judged against the band 0.010"
nohup $RUN --id E237.a0 --seed 0 --after E240.c2 --hyp "$H37" --pred "$P37" --rule "$R37" -- env LCR_DIST_DIR=$DIST $T e24_D7_j1exs0 0 1.0 e24_j1ex_s0 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E237.a1 --seed 1 --after E237.a0 --hyp "$H37" --pred "$P37" --rule "$R37" -- env LCR_DIST_DIR=$DIST $T e24_D7_j1exs1 1 1.0 e24_j1ex_s1 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E244.b0 --seed 0 --after E237.a1 --hyp "$H44" --pred "$P44" --rule "$R44" -- env P19_REC_CLAMP=8.0 $T e24_D7_j1cl8s0 0 1.0 e24_j1cl8_s0 > /dev/null 2>&1 &
sleep 3
H40="E240: the recall weight 3 gain of j1rc over the equal-budget j1eq is measured on four paired seeds with every headline metric (seed noise = measurement floor)"
P40="Q j1rc - j1eq = +0.006 (mean of 4 paired seeds), lanes found +0.06, seed SD of Q <= 0.004"
R40="Q a - c >= 0.005 and lanes found >= +0.04 with topo_F1 not lower by more than 0.01 and F1@1.0 not lower by more than 0.005 (decide.py --experiment E240)"
nohup $RUN --id E240.a3 --seed 3 --after E244.b0 --hyp "$H40" --pred "$P40" --rule "$R40" -- $T e24_D7_j1rcs3 3 3.0 e24_j1rc_s3 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.c3 --seed 3 --after E240.a3 --hyp "$H40" --pred "$P40" --rule "$R40" -- $T e24_D7_j1eqs3 3 1.0 e24_j1eq_s3 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.ev0 --after E240.c3 --hyp "$H40" --pred "reproduce the stored seeds 0 / 1: evaluate.py cards of e24_j1rc_s0/s1 and e24_j1eq_s0/s1 match the E219 numbers (Q 0.739 / 0.743 and 0.735 / 0.731)" --rule "layer 1: stored numbers reproduced within 0.002; then decide.py --experiment E240" -- bash -c 'for n in e24_j1rc_s0 e24_j1rc_s1 e24_j1eq_s0 e24_j1eq_s1; do bash tools/env/_py14.sh evaluate.py $n 2>&1 | tail -n 12; done' > /dev/null 2>&1 &
sleep 1
echo "queue: E240.a2 (running) -> c2 -> E237.a0 -> a1 -> E244.b0 -> E240.a3 -> c3 -> ev0"
