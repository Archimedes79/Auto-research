#!/usr/bin/env bash
# verify a commit tree from a git archive: the new self-tests, CPU, seconds, no training step.   usage: verify_tree.sh <tree>
set -u
TREE=$1
REPO=/mnt/c/Auto-research
V=$REPO/bev_fusion/temp/head_session_2026-11-01/verify
PY=~/miniconda3/envs/gcar/bin/python
rm -rf "$V"; mkdir -p "$V"
git -C $REPO archive "$TREE" bev_fusion/training bev_fusion/tests autogt | tar -x -C "$V"
mkdir -p "$V/bev_fusion/temp"
ln -s $REPO/bev_fusion/data "$V/bev_fusion/data"
ln -s $REPO/bev_fusion/models "$V/bev_fusion/models"
cd "$V/bev_fusion" || exit 1
export OMP_NUM_THREADS=2
fail=0
run() { local name=$1; shift; if out=$(nice -n 10 "$@" 2>&1); then echo "PASS  $name | $(echo "$out" | tail -1 | cut -c1-150)"; else echo "FAIL  $name | $(echo "$out" | tail -4 | tr '\n' ' ' | cut -c1-300)"; fail=1; fi; }
run "fusion-area loss"   $PY training/losses.py --fusion-area
run "startable selftest" $PY training/startable.py
run "fusion area, real"  $PY tests/test_fusion_area_real.py
# the t14 arm in the archive: the run folder / pinned autogt do not exist here, so only the static part is judged
$PY training/startable.py training/configs/t14_dc_fusion_s0.json 2>&1 | cut -c1-260 | sed 's/^/      /'
# unlink the symlinks before anything is removed
rm "$V/bev_fusion/data" "$V/bev_fusion/models"
echo "verify exit $fail"
exit $fail
