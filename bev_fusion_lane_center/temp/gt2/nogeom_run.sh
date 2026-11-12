cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e305_e_s0 0 0.1 nogeom >> temp/gt2/snapopt_nogeom.log 2>&1
nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e305_e_s1 1 0.1 nogeom >> temp/gt2/snapopt_nogeom.log 2>&1
echo NOGEOM_DONE >> temp/gt2/snapopt_nogeom.log
