cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for S in 0 1; do
  nice -n 5 bash tools/env/_py14.sh temp/gt2/thr_sweep.py e305_e_s$S $S 0.1 0.5 >> temp/gt2/thr_e305.log 2>&1
done
echo THR305_DONE >> temp/gt2/thr_e305.log
