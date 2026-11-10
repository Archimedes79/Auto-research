#!/bin/bash
cd /mnt/c/Auto-research || exit 1
P="bev_fusion/evaluation/evalsuite.json"
cat > bev_fusion/temp/cleanup/commit_hook_msg.txt <<'EOF'
evaluation/evalsuite.json: the project names its Runner ("runner": "runner.runner:Runner"), so the Evaluation always includes the class of bev_fusion/runner (Mirko 2026-11-10: "Evaluation soll immer diese Runnerklasse includieren"); the key is read by the Evaluation chat's load_runner

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>
EOF
python3 autoresearch/tools/check_git.py --paths $P || exit 1
python3 autoresearch/tools/commit_paths.py -F bev_fusion/temp/cleanup/commit_hook_msg.txt $P || exit 1
git log --oneline -1 | cut -c1-140
