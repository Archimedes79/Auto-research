#!/bin/bash
# exits when the last job of the chain has a final status or an earlier one failed
cd /mnt/c/Auto-research/bev_fusion
st() { python3 -c 'import json,sys;print(json.load(open("records/runs/%s/run.json" % sys.argv[1])).get("status"))' "$1" 2>/dev/null; }
while true; do
  a=$(st t26.base_eval); b=$(st t26.base_eval43); c=$(st t26.ref_maps_base)
  case "$a$b$c" in *failed*|*blocked*|*stalled*|*stopped*|*held*) break;; esac
  [ -e records/runs/t26.ref_maps_base/done ] && break
  sleep 120
done
echo "base_eval=$a base_eval43=$b ref_maps_base=$c"; date
