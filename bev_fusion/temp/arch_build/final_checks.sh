#!/usr/bin/env bash
# one-off: the last round after the bug hunt's fixes: dry run, clip equivalence (references of the old code kept), slots, gates
cd "$(dirname "$0")/../.." || exit 1
P=~/miniconda3/envs/gcar/bin/python
echo "=== dry run"
OMP_NUM_THREADS=4 $P temp/arch_build/dryrun.py 2>&1 | grep -v Warning | grep -E "^  (ok|FAIL)|parameters without|dry run|Error" | head -20
echo "=== equivalence (clips, CPU)"
for spec in "runs/t10_dual_s0 ckpt_00003000.pt base" "runs/t10_dual_long_s0 ckpt_00005030.pt hurdle"; do
  set -- $spec
  OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES= EQUIV_DEVICE=cpu EQUIV_RUN=$1 EQUIV_CKPT=$2 EQUIV_REF=temp/arch_build/ref_clip_$3.pt \
    $P tests/equivalence.py compare 2>&1 | grep -v Warning | tail -2 | cut -c1-200
done
echo "=== slots + self-tests"
bash temp/arch_build/run_tests.sh 2>&1 | tail -4
echo "=== gates"
OMP_NUM_THREADS=4 timeout 3000 bash tests/run_gates.sh 2>&1 | grep -E "^(PASS|FAIL)" | cut -c1-70
