cd /mnt/c/Auto-research/bev_fusion
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY=/home/ml/miniconda3/envs/gcar/bin/python
for m in object_head timed_head losses bev_trunk; do
  echo "=== $m"; s=$(date +%s)
  timeout 900 $PY training/$m.py 2>&1 | tail -6 | cut -c1-400
  echo "exit ${PIPESTATUS[0]}  $(( $(date +%s) - s )) s"
done
