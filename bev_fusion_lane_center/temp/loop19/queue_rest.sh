#!/bin/bash
# Re-queue after E244.b0 (23:30): E244 arm a (Poisson set loss, phd 0.65) seeds 0 / 1, then the E240 seed pair 3, then E244.b1 (clamp 8 m, seed 1).
# Why: the first requeue (requeue_gpu.sh) relaunched E240.a3 / c3 / ev0 while the killed launchers still had a fresh heartbeat; run.py refused the id reuse, E244.a0 then
# blocked on the stale E240.ev0. E240.ev0 is dropped (the stored seeds were evaluated by E240.ev1).
cd "$(dirname "$0")/../.."
RUN="python3 ../autoresearch/tools/run.py --gpu-max-mb 9000 --retries 0"
T="bash temp/loop19/train_j1.sh"
HA="E244 arm a: replacing the clamped hinge recall by the matching-free Poisson set likelihood (phd) gives more lanes found and Q than the equal-budget plain fine-tune j1eq, and Q above the recall-weight-3 fine-tune j1rc"
PA="Q +0.006 over j1eq (mean of 2 seeds; the card said +0.012, the referee asked to shrink), lanes found +0.06, R@1 far +0.05, Q against j1rc +0.003 (clause band 0.003), at tau 0.5"
RA="KEEP if Q a - c >= 0.010 with both seeds on the same side, lanes found >= +0.05, R@1_far >= +0.03, topo_F1 not lower by more than 0.005 and Q a - r >= 0.003; TIE inside the band (keep the plain fine-tune); a Q gain only against c is a 'form of the loss' result, no promotion"
H40="E240: the recall weight 3 gain of j1rc over the equal-budget j1eq is measured on four paired seeds with every headline metric (seed noise = measurement floor)"
P40="Q j1rc - j1eq = +0.006 (mean of 4 paired seeds), lanes found +0.06, seed SD of Q <= 0.004"
R40="Q a - c >= 0.005 and lanes found >= +0.04 with topo_F1 not lower by more than 0.01 and F1@1.0 not lower by more than 0.005 (decide.py --experiment E240)"
nohup $RUN --id E244.a0 --seed 0 --after E244.b0 --hyp "$HA" --pred "$PA" --rule "$RA" -- $T e24_D7_j1phds0 0 0.0 e24_j1phd_s0 --w rec=0 phd=0.65 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E244.a1 --seed 1 --after E244.a0 --hyp "$HA" --pred "$PA" --rule "$RA" -- $T e24_D7_j1phds1 1 0.0 e24_j1phd_s1 --w rec=0 phd=0.65 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.a3 --seed 3 --after E244.a1 --hyp "$H40" --pred "$P40" --rule "$R40" -- $T e24_D7_j1rcs3 3 3.0 e24_j1rc_s3 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E240.c3 --seed 3 --after E240.a3 --hyp "$H40" --pred "$P40" --rule "$R40" -- $T e24_D7_j1eqs3 3 1.0 e24_j1eq_s3 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E244.b1 --seed 1 --after E240.c3 --hyp "E244 arm b: lifting the recall clamp from 3 m to 8 m (recall weight 1) lifts lanes found of the plain fine-tune; seed 1" --pred "lanes found between j1eq (0.674) and j1rc (0.736), Q within +-0.005 of j1eq" --rule "as E244.b0" -- env P19_REC_CLAMP=8.0 $T e24_D7_j1cl8s1 1 1.0 e24_j1cl8_s1 > /dev/null 2>&1 &
sleep 1
echo "queued after E244.b0: E244.a0 -> a1 -> E240.a3 -> c3 -> E244.b1"
