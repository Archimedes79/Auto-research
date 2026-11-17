cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
NET=$1
bash tools/env/_py14.sh temp/gt2/snap_score.py ${NET}1 1 0.1 0.5 0.2 > temp/gt2/snapc2_${NET}1.log 2>&1
echo SNAPC2_DONE >> temp/gt2/snapc2_${NET}1.log
