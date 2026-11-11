#!/bin/bash
# scratch (E282): starts the CPU mini-pilot of card experiments/E282_zone-matching-node-loss section 3 through run.py: the cache once, then the four arms in parallel (2 threads each) behind it
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
HYP="E282: the node head trained on the sparse node target (Douglas-Peucker 0.3 m / 7.5 m) with the zone-matching existence loss delivers a thin row (one node per zone, any cell of the zone) that the per-cell sparse loss of E276 arm s cannot place; CPU mini-pilot on 601 TRAIN + 110 VAL frames, 2000 steps, one seed"
PRED="arm m: free_zone_exact 0.55 (0.20 - 0.80), free_zone_recall 0.75 (0.50 - 0.90), about 15 nodes per 100 m; control s at matched density free_zone_exact 0.25 (0.10 - 0.40); gain about +0.30; chance of GO about 3 in 10"
RULE="card E282 section 3, written before the run, implemented by qnode_pilot.py gate: INVALID if dense arm d fails the node gate (precision 0.75, recall 0.5); INCONCLUSIVE-control if s is not at 14.4 +- 3 nodes per 100 m on VAL; S-PASSES if s passes the floors; GO if m passes all floors (free recall >= d - 0.10, free exact >= 0.50, off-zone <= 0.15, 8 - 22 nodes per 100 m, fixed recall >= s - 0.05, gain over s at matched density >= 0.15 with bootstrap lo90 > 0.05, winners off the GT cell >= 0.25); else STOP-A / STOP-B"
python3 ../autoresearch/tools/run.py --id E282.p_cache --no-card -- bash tools/env/_py14.sh training/train/qnode_pilot.py cache &
for a in d s m m5; do
  python3 ../autoresearch/tools/run.py --id E282.p_$a --hyp "$HYP (arm $a)" --pred "$PRED" --rule "$RULE" --seed 0 --after E282.p_cache --ram-mb 3000 -- bash tools/env/_py14.sh training/train/qnode_pilot.py train --arm $a --steps 2000 &
done
wait
echo "E282 pilot launcher: all runs ended"
