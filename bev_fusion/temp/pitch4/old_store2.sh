# how the Evaluation treats a stored evaluator store whose hash is not the current one (CPU, read-only)
cd "$(git rev-parse --show-toplevel)/bev_fusion"
PY=$HOME/miniconda3/envs/gcar/bin/python
export PYTHONPATH="$(cd .. && pwd)"
echo "== Evaluation methods:"; (cd ../autoresearch && AR_PROJECT=../bev_fusion $PY -m Evaluation methods 2>&1 | cut -c1-170 | head -30)
echo "== Runner methods:"; $PY -m runner methods 2>&1 | cut -c1-170 | head -12
