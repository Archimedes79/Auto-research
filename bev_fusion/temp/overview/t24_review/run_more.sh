cd /mnt/c/Auto-research/bev_fusion || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/gcar/bin/python
for t in tests/test_student_chain.py tests/test_host_batch.py tests/test_slots.py; do
  echo "=== $t"
  nice -n 12 timeout 900 $PY $t > temp/overview/t24_review/last.log 2>&1; echo "exit $?"
  grep -E "FAIL|passed|all ok|FAILED|Error|Traceback" temp/overview/t24_review/last.log | head -n 8
  tail -n 2 temp/overview/t24_review/last.log | cut -c1-200
done
echo done
