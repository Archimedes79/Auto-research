cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
nice -n 10 bash tools/env/_py14.sh evaluation/jmaneuver.py score --run e308_t_s0 --seed 0 --variants plain p0.5 p0.2 p0.1 oracle oracle+fit --rows > temp/jr/jman_run.log 2>&1
echo JMAN_DONE >> temp/jr/jman_run.log
