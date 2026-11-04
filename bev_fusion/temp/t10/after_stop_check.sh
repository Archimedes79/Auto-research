#!/bin/bash
# after the stop of t10_dual_long_s0: wait (max 4 minutes) until the trainer process is gone, then show the run record, the chained rows, the GPU and the memory
cd "$(dirname "$0")/../.." || exit 1
for i in $(seq 1 80); do
  pgrep -f "t10_dual_long_s0/code/train.py" > /dev/null || break
  sleep 3
done
date +%H:%M:%S
echo "trainer processes left: $(pgrep -f 't10_dual_long_s0/code/train.py' | wc -l)"
nvidia-smi.exe --query-gpu=memory.used,utilization.gpu --format=csv,noheader
free -m | sed -n 2p
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/ledger.py tail 2>&1 | grep -E "t10.dual_long|maps_lfl_long" | cut -c1-230
echo "---- run record:"
python3 - <<'EOF'
import json
for d in ("t10.dual_long_s0", "t10.dual_long_s0.eval", "pitch.maps_lfl_long"):
    try:
        j = json.load(open("records/runs/%s/run.json" % d))
    except Exception as e:
        print(d, "no record", e)
        continue
    print(d, {k: j.get(k) for k in ("status", "exit", "ended", "notes", "commit") if k in j})
EOF
