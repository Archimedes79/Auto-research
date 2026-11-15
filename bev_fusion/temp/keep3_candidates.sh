#!/bin/bash
# scratch: keep3 state and candidates of bev_fusion (read-only)
cd /mnt/c/Auto-research || exit 1
export AR_PROJECT=../bev_fusion
echo "=== show"; python3 autoresearch/tools/keep3.py show 2>&1 | cut -c1-230
echo; echo "=== candidates"; python3 autoresearch/tools/keep3.py candidates --top 25 2>&1 | cut -c1-230
echo; python3 - <<'EOF'
import json
k = json.load(open("/mnt/c/Auto-research/bev_fusion/records/keep3.json"))
print({x: k[x] for x in k if x not in ("entries", "about")})
EOF
