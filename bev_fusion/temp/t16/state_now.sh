#!/bin/bash
# scratch: state of the t18 runs (run.json status, last log lines, queue)
cd /mnt/c/Auto-research/bev_fusion/records/runs || exit 1
date +%T
for r in t18.fix t18.fix.eval showcase.gpu; do
  echo "== $r"
  python3 - "$r" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1] + "/run.json"))
print({k: d.get(k) for k in ("status", "exit", "started", "ended", "commit", "heartbeat")})
EOF
  tail -n 3 "$r/log.txt" 2>/dev/null | cut -c1-200
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
