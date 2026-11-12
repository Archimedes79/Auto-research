cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for S in 0 1; do
  nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e312_p_s$S $S 0.05 plain >> temp/gt2/thr_e312.log 2>&1
  nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e312_p_s$S $S 0.02 plain >> temp/gt2/thr_e312.log 2>&1
done
echo THR312_DONE >> temp/gt2/thr_e312.log
