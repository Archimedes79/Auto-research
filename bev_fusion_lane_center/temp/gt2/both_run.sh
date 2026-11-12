cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
until grep -q NOGEOM_DONE temp/gt2/snapopt_nogeom.log 2>/dev/null; do sleep 15; done
nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e305_e_s0 0 0.1 both >> temp/gt2/snapopt_both.log 2>&1
nice -n 5 bash tools/env/_py14.sh temp/gt2/snapopt_sweep.py e305_e_s0 0 0.05 plain >> temp/gt2/snapopt_both.log 2>&1
echo BOTH_DONE >> temp/gt2/snapopt_both.log
