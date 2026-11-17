#!/bin/bash
# U1 scratch: the importers' tests of jtoken_eval with the edited module preloaded (u1_run_with.py); the subprocess-based `selftests` modes are run as their subcommands
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
W="nice -n 10 bash tools/env/_py14.sh u1_run_with.py"
R=$(pwd)
run() { echo "=== $*"; $W "$@" 2>&1 | tail -n 4; echo "rc ${PIPESTATUS[0]}"; }
run training/grid/test_jtoken.py prior teacher
run training/grid/test_jmap_train.py one "$R/training/train/jmap_train.py" "$R/evaluation/jmap_eval.py"
run training/grid/test_jmap_train.py real
run training/grid/test_jpast_train.py one "$R/training/train/jpast_train.py" "$R/evaluation/jpast_eval.py"
run training/grid/test_jpast_train.py real
run evaluation/tests/test_token_readout.py
run tools/jarm.py selftest
