#!/bin/bash
# Loop 20, iteration 4: E255 arm a (Poisson set loss, full schedule, exact distance map = loader default), chained behind the pair-3 trainings.
cd "$(dirname "$0")/../.."
RUN="python3 ../autoresearch/tools/run.py --gpu-max-mb 9000 --retries 0 --gpu"
HA="E255: replacing the clamped hinge recall by the matching-free Poisson set likelihood does not lift Q, lanes found or R@1 far of the j1 fine-tune at a matched vertex budget on the same exact distance map"
PA="Q a - c = -0.030 at 14.3 vertices per 100 m, lanes found -0.03"
RA="CUT if Q a - c <= -0.010 (one seed is enough); KEEP only if Q a - c >= +0.010, lanes found >= +0.05, R@1_far >= +0.03 and topo_F1 loss <= 0.005 on both seeds; TIE otherwise (take the plain fine-tune); matched-vertex procedure of the card"
nohup $RUN --id E255.a0 --seed 0 --after E240.c3 --hyp "$HA" --pred "$PA" --rule "$RA" -- bash temp/loop19/train_j1.sh e25_D7_j1phd2s0 0 0.0 e25_j1phd2_s0 --w rec=0 phd=0.65 > /dev/null 2>&1 &
sleep 1; echo queued E255.a0 after E240.c3
