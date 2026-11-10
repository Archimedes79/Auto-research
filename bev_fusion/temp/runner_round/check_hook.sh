#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
$PY - <<'EOF'
import sys
sys.path[:0] = ["/mnt/c/Auto-research/autoresearch"]
from evalsuite.api import Evaluation
ev = Evaluation("/mnt/c/Auto-research/bev_fusion")
print("runner class:", type(ev.runner).__module__ + "." + type(ev.runner).__name__)
print("sets:", list(ev.list_sets())[:4])
EOF
echo "exit $?"
$PY tests/test_evaluation_api.py 2>&1 | tail -4 | cut -c1-200
echo "exit ${PIPESTATUS[0]}"
