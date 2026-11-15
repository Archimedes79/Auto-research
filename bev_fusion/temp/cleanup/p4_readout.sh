#!/usr/bin/env bash
# scratch: t25 go / no-go read-out on the stored P4 checkpoint (runs/t21_pair4_s0, ckpt_00001500, EMA), 246 VAL frames, CPU, the live training code (no code/ snapshot in the
# variant folders: the run's own snapshot imports the old autogt package, D81), GT v13 from the trainer cache. The stored run is never written: each variant is its own folder.
#   a  as is (radar_window 0.5)           b  --control zero_radar           c  radar_window 0.09 (the newest sweep only, no training)
# side job on the shared box: nice 15, 3 threads, 1 read-ahead worker.
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
PY="${PY:-$HOME/miniconda3/envs/gcar/bin/python}"
P4=runs/t21_pair4_s0
BASE=temp/cleanup/p4_readout
until grep -q "split val rc=0" temp/cleanup/gtcache_v13.log; do sleep 20; done        # the val GT is in the cache
for v in a b c; do
  mkdir -p "$BASE/$v"
  "$PY" - "$P4/config.json" "$BASE/$v/config.json" "$v" <<'EOF'
import json, sys
src, dst, v = sys.argv[1:4]
cfg = json.load(open(src))
if v == "c":
    cfg["radar_window"] = 0.09
cfg["name"] = "p4_readout_" + v
json.dump(cfg, open(dst, "w"), indent=1)
EOF
done
for v in a b c; do
  ctl=""
  [ "$v" = "b" ] && ctl="--control zero_radar"
  echo "== $(date '+%F %T') variant $v"
  nice -n 15 "$PY" evaluation/bev_suite.py "$BASE/$v" --ckpt-dir "$P4" --device cpu --workers 1 $ctl eval --n 246 > "$BASE/$v.log" 2>&1
  echo "== $(date '+%F %T') variant $v rc=$?"
done
echo "== all done $(date '+%F %T')"
