#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion || exit 1
python3 temp/cleanup/unused_scan.py runner/runner.py runner/__main__.py runner/__init__.py tests/test_runner_class.py tests/test_runner_real.py tests/test_suites_conform.py tests/test_bev_free.py evaluation/bev_free.py
