#!/bin/bash
cd /mnt/c/Auto-research || exit 1
echo "=== tests"; bash bev_fusion/temp/speed_tools/run_tests.sh
echo "=== README / run_gates blobs"; python3 bev_fusion/temp/speed_tools/readme_rows3.py
for f in bev_fusion/training/README.md bev_fusion/tests/README.md bev_fusion/tests/run_gates.sh; do
  n=$(echo $f | sed 's#bev_fusion/##; s#/#_#; s#\.#_#')
  git show HEAD:$f > /tmp/head_blob.md
  echo "== $f: blob vs HEAD changed lines: $(diff /tmp/head_blob.md bev_fusion/temp/blob3_$n | grep -c '^[<>]')"
done
FILES="bev_fusion/training/losses.py bev_fusion/training/object_head.py bev_fusion/training/timed_head.py bev_fusion/training/splat.py bev_fusion/training/train.py bev_fusion/evaluation/bev_adapter.py bev_fusion/tests/test_hygiene.py bev_fusion/tests/test_lr_groups.py bev_fusion/tests/toy_train.py"
echo "=== status of my files"; git status --short $FILES
echo "=== hunks vs HEAD per file"; for f in $FILES; do printf "%-48s %s\n" $f "$(git diff HEAD -U0 -- $f | grep -c '^@@')"; done
echo "=== check_git"; python3 autoresearch/tools/check_git.py --paths $FILES 2>&1 | tail -3
echo "=== commit ($1)"
python3 autoresearch/tools/commit_paths.py -F temp/msg_hygiene_lr.txt $FILES \
  --blob bev_fusion/training/README.md=bev_fusion/temp/blob3_training_README_md \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/blob3_tests_README_md \
  --blob bev_fusion/tests/run_gates.sh=bev_fusion/temp/blob3_tests_run_gates_sh $1 2>&1 | tail -16
