#!/usr/bin/env bash
# one-off: the training clip on the GPU in bf16, the frames of the traceback
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=4 DBG_DEVICE=cuda DBG_AMP=bf16
timeout 1500 ~/miniconda3/envs/gcar/bin/python temp/arch_build/dbg_clip.py 2>&1 | grep -v Warning | grep -E 'File "/mnt|Error|flip ' | tail -6
