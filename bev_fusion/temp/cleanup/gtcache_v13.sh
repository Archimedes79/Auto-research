#!/usr/bin/env bash
# scratch: prefill the trainer GT cache (data/cache/gt) for the v13 GT signature, val first, then train (D83 / D91 prerequisite, t25 needs --expect-gt a107e0b4e8)
# side job on the shared box: nice 15, 2 workers, 1 thread each. Resumable: existing files are skipped by the writer.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
for split in val train; do
  echo "== $(date '+%F %T') split $split"
  nice -n 15 "$PY" training/gt_cache_build.py ~/data/nuscenes_full v1.0-trainval data/cache/gt --split "$split" --workers 2
  rc=$?
  echo "== $(date '+%F %T') split $split rc=$rc"
  [ "$rc" -eq 0 ] || exit "$rc"
done
echo "== all done $(date '+%F %T')"
