# would the CURRENT code reuse the result sets of chain_t20 / chain2_t20 (made before the rename)? Runner.plan = dry run, nothing runs (CPU, read-only)
cd "$(git rev-parse --show-toplevel)/bev_fusion"
PY=$HOME/miniconda3/envs/gcar/bin/python
export PYTHONPATH="$(cd .. && pwd):$(cd .. && pwd)/autoresearch" AR_PROJECT=.
for n in chain_t20 chain2_t20; do
  m=$(ls runs/$n/eval/results/*/manifest.json | head -1)
  echo "== $n identity in the stored manifest:"; $PY - "$m" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(" ", json.dumps(d["identity"], sort_keys=True)[:600])
EOF
  echo "-- Runner.plan $n small.val:"
  $PY -m runner Runner.plan $n small.val 2>&1 | cut -c1-320 | head -14
done
