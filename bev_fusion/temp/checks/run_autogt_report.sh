#!/bin/bash
# GT evaluation of all mini sequences: stores the GT, writes the PDF report (with the radar hold-out test); log next to it
cd "$(dirname "$0")/../.." || exit 1
mkdir -p autogt/reports
~/miniconda3/envs/gcar/bin/python -m autogt report --version v1.0-mini --holdout > autogt/reports/report_run.log 2>&1
