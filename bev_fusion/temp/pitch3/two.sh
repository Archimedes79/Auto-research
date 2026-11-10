cd "$(git rev-parse --show-toplevel)/bev_fusion"
PY=$HOME/miniconda3/envs/gcar/bin/python
$PY tests/test_suites_conform.py 2>&1 | grep -v -i warn | tail -5 | cut -c1-400
echo ======
$PY tests/test_runner_class.py 2>&1 | grep -v -i warn | tail -8 | cut -c1-500
