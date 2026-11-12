cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
# waits for the snap confirmation to finish (CPU is shared with the trainings), then the wiring sweep on E305 (the best network so far), thr 0.1 / 0.05
until grep -q SNAPC_DONE temp/gt2/snapc_e305_e_s1.log 2>/dev/null && grep -q SNAPC_DONE temp/gt2/snapc_e287_p2_s1.log 2>/dev/null; do sleep 15; done
for S in 0 1; do
  (for PA in "6 45" "6 60" "8 60"; do set -- $PA; bash tools/env/_py14.sh temp/gt2/wire_sweep.py e305_e_s$S $S $1 $2 0.1 >> temp/gt2/wire_e305_s$S.log 2>&1; done
   bash tools/env/_py14.sh temp/gt2/wire_sweep.py e305_e_s$S $S 4 45 0.05 0.02 >> temp/gt2/wire_e305_s$S.log 2>&1) &
done
wait
echo WIRE_DONE >> temp/gt2/wire_e305_s1.log
