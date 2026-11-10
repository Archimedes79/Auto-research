#!/bin/bash
# scratch (Image Encoder, clean-base round): the tests of my area on EXACTLY what will be committed: an export of HEAD (git archive, no other chat's uncommitted edits) with my blobs and my clean files laid over it.
# The critic asked for this: my first test run used the working tree, which carries other chats' edits in train.py / startable.py.
cd /mnt/c/Auto-research || exit 1
CB=/tmp/cb_t22
rm -rf $CB; mkdir -p $CB
git archive HEAD bev_fusion/training bev_fusion/tests bev_fusion/evaluation bev_fusion/runner autoresearch GtGenerator 2>/dev/null | tar -x -C $CB
ls $CB | tr '\n' ' '; echo
H=bev_fusion/temp/cleanup/headcopy
for f in bev_fusion/training/train.py bev_fusion/training/startable.py; do cp $H/$f $CB/$f; done
for f in bev_fusion/training/lane_branch.py bev_fusion/training/feature_cache.py bev_fusion/training/model.py bev_fusion/training/interface.py bev_fusion/tests/test_student_chain.py; do cp $f $CB/$f; done
rm -f $CB/bev_fusion/training/configs/t22_B2_s0.json
# data the code looks for through links of the real tree (caches, runs, models): point them at the real folders
for d in data runs models temp; do [ -e $CB/bev_fusion/$d ] || ln -s /mnt/c/Auto-research/bev_fusion/$d $CB/bev_fusion/$d; done
cd $CB/bev_fusion || exit 1
export PYTHONPATH="$CB" OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
PY="$HOME/miniconda3/envs/gcar/bin/python"
LOG=/mnt/c/Auto-research/bev_fusion/temp/cleanup/my_tests_on_blobs.log
: > $LOG
run() { local name="$1"; shift; local t0=$(date +%s); local out; out=$("$@" 2>&1); local rc=$?
  echo "$( [ $rc = 0 ] && echo PASS || echo FAIL )  $name  ($(( $(date +%s) - t0 )) s)  | $(echo "$out" | grep -v -i warn | tail -2 | tr '\n' ' ' | cut -c1-200)" >> $LOG
  [ $rc = 0 ] || echo "$out" | tail -30 >> $LOG.fail; }
rm -f $LOG.fail
run "student chain (with the new K)" $PY tests/test_student_chain.py
run "startable selftest"             $PY training/startable.py
run "interface selftest"             $PY training/interface.py
run "lane_branch selftest"           $PY training/lane_branch.py
run "feature_cache selftest"         $PY training/feature_cache.py
run "module probe"                   $PY evaluation/bev_modules.py --selftest
run "train guards"                   $PY tests/test_train_guards.py
echo "ALL DONE $(date +%H:%M:%S)" >> $LOG
