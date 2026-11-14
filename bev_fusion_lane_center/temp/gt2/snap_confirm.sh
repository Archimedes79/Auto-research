cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
NET=$1
for S in 0 1; do
  bash tools/env/_py14.sh temp/gt2/snap_score.py $NET$S $S 0.1 0.5 0.2 > temp/gt2/snapc_$NET$S.log 2>&1
done
echo SNAPC_DONE >> temp/gt2/snapc_${NET}1.log
