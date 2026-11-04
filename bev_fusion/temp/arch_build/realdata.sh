#!/usr/bin/env bash
# one-off: the real-data checks of the bug-fixed framework, one after the other (each loads the trainval index, ~8-10 GB host memory)
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=4
P=~/miniconda3/envs/gcar/bin/python
echo "=== 1 precompute on its own"
$P training/precompute.py check runs/t10_dual_long_s0/config.json --n 12 2>&1 | grep -v Warning | tail -2
$P training/precompute.py frames runs/t10_dual_long_s0/config.json --scene scene-0003 --out temp/arch_build/precompute_scene0003 2>&1 | grep -v Warning | tail -2
ls temp/arch_build/precompute_scene0003 | head -3; du -sh temp/arch_build/precompute_scene0003
echo "=== 2 dry run"
$P temp/arch_build/dryrun.py 2>&1 | grep -v Warning | grep -E "^  (ok|FAIL)|parameters without|dry run|Error" | head -20
echo "=== 3 evaluation path"
bash temp/arch_build/evalrun.sh
echo "=== 4 train selftest"
timeout 1800 $P training/train.py 2>&1 | grep -v Warning | tail -3
