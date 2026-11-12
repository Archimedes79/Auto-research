cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for S in 0 1; do
  bash tools/env/_py14.sh temp/gt2/thr_sweep.py e287_p2_s$S $S 0.5 0.3 0.2 0.1 > temp/gt2/thr_p2_s$S.log 2>&1
done
echo THRP2_DONE >> temp/gt2/thr_p2_s1.log
