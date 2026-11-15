#!/bin/bash
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=
bash tools/env/_py14.sh jends_train.py --help 2>&1 | grep -E "geom-w|--unc" | head -4
bash tools/env/_py14.sh jends_train.py --name x --unc 2>&1 | tail -n 1
