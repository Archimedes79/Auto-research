#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/runner/__init__.py bev_fusion/runner/README.md bev_fusion/tests/test_runner_class.py"
cat > bev_fusion/temp/cleanup/commit_init_msg.txt <<'EOF'
runner/__init__.py: `from runner import Runner` works (the folder was a bare namespace package, the class only importable as runner.runner.Runner); README row and example; the toy test uses the short import

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_init_msg.txt $P || exit 1
git log --oneline -1 | cut -c1-140
