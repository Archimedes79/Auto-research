#!/bin/bash
# scratch: static scan of my area (unused imports/variables, stale names, structure check)
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=~/miniconda3/envs/gcar/bin/python
echo "=== pyflakes / vulture available?"
$PY -m pyflakes --version 2>&1 | head -1
$PY -m vulture --version 2>&1 | head -1
$PY -m ruff --version 2>&1 | head -1
echo "=== pyflakes on my files"
$PY -m pyflakes runner/runner.py runner/__main__.py runner/__init__.py tests/test_runner_class.py tests/test_runner_real.py tests/test_suites_conform.py tests/test_bev_free.py 2>&1 | head -30
echo "=== stale names in my docs and code"
grep -n -i "runsuite\|RunJobs\|runner_api\|DECISIONS\.md\|FROZEN\|engine\.py" runner/*.py runner/README.md records/jobs/README.md records/jobs/*/README.md tests/test_runner_class.py tests/test_runner_real.py tests/test_suites_conform.py 2>/dev/null | cut -c1-200
echo "=== structure check (bev_fusion)"
cd /mnt/c/Auto-research && AR_PROJECT=../bev_fusion python3 autoresearch/tools/check_structure.py 2>&1 | tail -12 | cut -c1-200
