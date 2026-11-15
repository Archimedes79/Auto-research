#!/bin/bash
# build the total report (CPU) and its single-file version
cd "$(dirname "$0")/../.."
export BFLC_ROOT=$PWD
export CUDA_VISIBLE_DEVICES=""
bash tools/env/_py14.sh ../temp/jr/build_total_report.py && python3 temp/jr/make_single_total.py
