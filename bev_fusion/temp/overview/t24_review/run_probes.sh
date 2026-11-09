cd /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=$HOME/miniconda3/envs/gcar/bin/python
echo "=== identity old (HEAD) vs new (working tree)"
nice -n 12 $PY probe_ident.py /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review/old /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review/ident_old.pt 2>&1 | tail -n 5
nice -n 12 $PY probe_ident.py /mnt/c/Auto-research/bev_fusion /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review/ident_new.pt 2>&1 | tail -n 5
nice -n 12 $PY probe_cmp.py /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review/ident_old.pt /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review/ident_new.pt 2>&1 | tail -n 15
echo "=== pull probes"
nice -n 12 $PY probe_pull.py 2>&1 | tail -n 30
echo "=== mutants of tests/test_lane_pull.py"
for m in m1 m2 m3; do
  echo "--- $m"
  (cd /mnt/c/Auto-research/bev_fusion/temp/overview/t24_review/$m && nice -n 12 $PY tests/test_lane_pull.py 2>&1 | grep -E "FAIL|all ok|FAILED|Error|Traceback" | head -n 12)
done
