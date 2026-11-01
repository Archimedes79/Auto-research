#!/bin/bash
# E244 arm a (Poisson set loss): two seeds after the E240 chain; phd weight 0.65 from the gradient calibration (temp/loop19/calib_phd.py: median |g rec| / |g phd| = 0.655), rec = 0
cd "$(dirname "$0")/../.."
RUN="python3 ../autoresearch/tools/run.py --gpu-max-mb 9000 --retries 0"
H="E244 arm a: replacing the clamped hinge recall by the matching-free Poisson set likelihood (phd) gives more lanes found and Q than the equal-budget plain fine-tune j1eq, and Q above the recall-weight-3 fine-tune j1rc"
P="Q +0.006 over j1eq (mean of 2 seeds; the card said +0.012, the referee asked to shrink), lanes found +0.06, R@1 far +0.05, Q against j1rc +0.003 (clause band 0.003), at tau 0.5"
R="KEEP if Q a - c >= 0.010 with both seeds on the same side, lanes found >= +0.05, R@1_far >= +0.03, topo_F1 not lower by more than 0.005 and Q a - r >= 0.003; TIE inside the band (keep the plain fine-tune); a Q gain only against c is a 'form of the loss' result, no promotion"
nohup $RUN --id E244.a0 --seed 0 --after E240.ev0 --hyp "$H" --pred "$P" --rule "$R" -- bash temp/loop19/train_j1.sh e24_D7_j1phds0 0 0.0 e24_j1phd_s0 --w rec=0 phd=0.65 > /dev/null 2>&1 &
sleep 3
nohup $RUN --id E244.a1 --seed 1 --after E244.a0 --hyp "$H" --pred "$P" --rule "$R" -- bash temp/loop19/train_j1.sh e24_D7_j1phds1 1 0.0 e24_j1phd_s1 --w rec=0 phd=0.65 > /dev/null 2>&1 &
sleep 1
echo "queued E244.a0 -> a1 after E240.ev0"
