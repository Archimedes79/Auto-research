#!/bin/bash
# Loop 20: E253 timing repeat on the quiet machine (the BEV chat's t1_r4 training has ended, 07:19).
cd "$(dirname "$0")/../.."
H="E253: j1 (192 / 96) is not a speed variant: ms per frame of j1 and the champion f1 agree within the 1.0 ms band on a quiet machine; recall costs about +2.5 ms"
P="ms per frame j1 - f1 (a - c) = -0.3 ms (tolerance 1.0 ms)"
R="session valid only with at least 5 valid replicates per configuration, replicate SD <= 0.6 ms in at least 5 of 6 configurations and stored values reproduced within 1.0 ms; then j1 - f1 <= -1.0 ms = KEEP, else TIE (analyse_e253.py)"
nohup python3 ../autoresearch/tools/run.py --id E253.t --gpu --hyp "$H" --pred "$P" --rule "$R" -- bash temp/loop20/run_e253.sh > /dev/null 2>&1 &
sleep 3
echo launched
