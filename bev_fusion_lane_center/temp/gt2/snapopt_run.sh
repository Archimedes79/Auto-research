cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
# E305 seed 0, thr 0.1: the control of the one-piece wiring and the two guards, one after the other (1 CPU thread)
for O in all guard dedup guard+dedup; do
  nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e305_e_s0 0 0.1 $O >> temp/gt2/snapopt_e305_s0.log 2>&1
done
echo SNAPOPT_DONE >> temp/gt2/snapopt_e305_s0.log
