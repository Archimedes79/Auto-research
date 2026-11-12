cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
for S in 0 1; do
  for R in e312_p_s$S e298_w_s$S; do
    echo "== $R" >> temp/day10/splits_driven.log
    bash tools/env/_py14.sh jsplits.py score --run $R 2>&1 | grep -E "pair recall  driven: |entry all|exit all" >> temp/day10/splits_driven.log
  done
done
echo SPLITS_DONE >> temp/day10/splits_driven.log
