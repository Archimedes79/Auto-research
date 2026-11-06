#!/usr/bin/env bash
# one-off: the training clip on the GPU (bf16), new code and the code before the build, blocking launches (exact line of an assert)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=4 CUDA_LAUNCH_BLOCKING=1 DBG_DEVICE=cuda DBG_AMP=bf16
P=~/miniconda3/envs/gcar/bin/python
for code in training temp/arch_build/old; do
  echo "=== $code"
  DBG_CODE=$code timeout 1500 $P temp/arch_build/dbg_clip.py 2>&1 | grep -v Warning | grep -E "code:|flip|Error|error|File \"/mnt" | tail -8
done
