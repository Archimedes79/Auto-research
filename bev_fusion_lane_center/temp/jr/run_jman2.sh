cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
nice -n 10 bash tools/env/_py14.sh evaluation/jmaneuver.py score --run e305_e_s0 --seed 0 --variants plain p0.5 p0.1 p0.5s p0.1s oracle oracle+fit > temp/jr/jman_e305.log 2>&1
echo JMAN2_DONE >> temp/jr/jman_e305.log
