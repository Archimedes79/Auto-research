#!/usr/bin/env bash
# scratch: the critic's four surviving mutants (bev_fusion/temp/t25_review/mut_run2.py, read only) against the extended test; logs in temp/t25/crit/
cd /mnt/c/Auto-research/bev_fusion || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH=/mnt/c/Auto-research
mkdir -p temp/t25/crit
for m in "$@"; do
  nice -n 15 "$HOME/miniconda3/envs/gcar/bin/python" temp/t25_review/mut_run2.py "$m" > "temp/t25/crit/$m.log" 2>&1
  echo "== $m: $(grep -c 'FAIL' temp/t25/crit/$m.log) FAIL lines; $(grep 'FAIL' temp/t25/crit/$m.log | head -2 | cut -c1-150 | tr '\n' '|')"
done
echo done
