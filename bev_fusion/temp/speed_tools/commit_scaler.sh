#!/bin/bash
cd /mnt/c/Auto-research || exit 1
echo "=== tests"; bash bev_fusion/temp/speed_tools/run_tests.sh
echo "=== README rows"; python3 bev_fusion/temp/speed_tools/readme_rows2.py
for f in bev_fusion/training/README.md bev_fusion/tests/README.md; do
  n=$(echo $f | sed 's#bev_fusion/##; s#/#_#; s#\.#_#')
  git show HEAD:$f > /tmp/head_blob.md
  echo "== $f: blob vs HEAD changed lines: $(diff /tmp/head_blob.md bev_fusion/temp/blob2_$n | grep -c '^[<>]')"
done
echo "=== check_git"
python3 autoresearch/tools/check_git.py --paths bev_fusion/training/train.py bev_fusion/runner/bev_runner.py bev_fusion/tests/test_grad_scaler.py bev_fusion/tests/run_gates.sh 2>&1 | tail -3
echo "=== hunks vs HEAD (train.py, runner, run_gates):"
git diff HEAD --stat -- bev_fusion/training/train.py bev_fusion/runner/bev_runner.py bev_fusion/tests/run_gates.sh | tail -4
echo "=== commit ($1)"
python3 autoresearch/tools/commit_paths.py -F temp/msg_fp16_scaler.txt \
  bev_fusion/training/train.py bev_fusion/runner/bev_runner.py bev_fusion/tests/test_grad_scaler.py bev_fusion/tests/run_gates.sh \
  --blob bev_fusion/training/README.md=bev_fusion/temp/blob2_training_README_md \
  --blob bev_fusion/tests/README.md=bev_fusion/temp/blob2_tests_README_md $1 2>&1 | tail -12
