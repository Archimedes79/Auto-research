#!/usr/bin/env bash
# a diagnostic run (no card, no verdict) through run.py: waits for a GPU slot and for the GPU memory in use to fall below 3000 MB. Run it in the foreground of a shell that stays alive.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id diag.t10_dual_s0.gen --no-card --no-git --gpu --gpu-max-mb 3000 -- bash temp/pitch/gen_eval.sh > runs/diag_t10_dual_s0_gen.launcher.log 2>&1
echo "launcher exit $?"
