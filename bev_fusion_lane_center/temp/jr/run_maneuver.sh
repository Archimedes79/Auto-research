cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
bash tools/env/_py14.sh ../jr/maneuver.py e308_t_s0 0 > temp/jr/maneuver_run.log 2>&1
echo MANEUVER_DONE >> temp/jr/maneuver_run.log
