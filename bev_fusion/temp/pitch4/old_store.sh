# a stored run made before the rename: do its old result sets and evaluator stores still LOAD, and what does the Evaluation say about their identity? (CPU, read-only)
cd "$(git rev-parse --show-toplevel)/bev_fusion"
PY=$HOME/miniconda3/envs/gcar/bin/python
export PYTHONPATH="$(cd .. && pwd)"
run=runs/t20_base_s0
echo "== result sets stored under $run/eval/results:"; ls $run/eval/results 2>/dev/null | head
echo "== evaluator stores under $run/eval/suite/quant:"; ls $run/eval/suite/quant 2>/dev/null | head
echo "== identity NOW (adapter hash in the record):"
$PY - <<'EOF'
import os, sys, json
sys.path[:0] = [os.getcwd(), os.path.join(os.getcwd(), "..", "autoresearch")]
from evaluation import bev_eval
idn = bev_eval.identity("runs/t20_base_s0")
print("  identity now:", {k: idn[k] for k in ("run", "checkpoint", "adapter")})
import glob
for m in sorted(glob.glob("runs/t20_base_s0/eval/results/*/manifest.json")):
    d = json.load(open(m))
    ident = d.get("identity") or d.get("prediction_identity") or {}
    print("  stored", os.path.basename(os.path.dirname(m)), "adapter:", ident.get("adapter"), "| keys:", sorted(d)[:6])
for m in sorted(glob.glob("runs/t20_base_s0/eval/suite/quant/*/manifest.json"))[:4]:
    d = json.load(open(m))
    print("  quant store", os.path.basename(os.path.dirname(m)), "evaluator_hash:", (d.get("describe") or d.get("source") or {}).get("evaluator_hash", d.get("evaluator_hash")))
EOF
echo "== the Runner lists them:"
$PY -m runner Runner.list_results run=t20_base_s0 2>&1 | tail -12 | cut -c1-220
