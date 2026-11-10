#!/bin/bash
cd "$(dirname "$0")/../.."
PY="$HOME/miniconda3/envs/gcar/bin/python"
OMP_NUM_THREADS=4 $PY -m cProfile -o temp/pitch/prof/eval.prof evaluation/bev_suite.py temp/pitch/prof/t16_full_s0 --device cuda --workers 2 --ckpt-dir runs/t16_full_s0 --curves --fusion --velocity --out temp/pitch/prof/t16_full_s0/suite eval --n 40 2>&1 | tail -4
cat > temp/pitch/prof/show.py <<'PYEOF'
import io
import pstats
s = io.StringIO()
p = pstats.Stats("temp/pitch/prof/eval.prof", stream=s)
p.sort_stats("cumulative").print_stats(70)
print(s.getvalue()[:12000])
PYEOF
$PY temp/pitch/prof/show.py > temp/pitch/prof/top.txt
echo "== prof done"
