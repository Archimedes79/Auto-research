#!/bin/bash
# scratch: README rows of the t24 unit. configs/README and training/README are clean in git; tests/README has another chat's uncommitted hunk: the working tree gets the row too, the commit uses a blob = HEAD + this row only.
cd /mnt/c/Auto-research/bev_fusion || exit 1
O=temp/overview
# configs/README: after the t23 row
n=$(grep -n '^| `t23_lane_bce.json`' training/configs/README.md | head -1 | cut -d: -f1)
[ -n "$n" ] && sed -i "${n}r $O/row_configs.txt" training/configs/README.md && echo "configs README row after line $n"
# training/README: extend the lane_branch.py row before its closing pipe
m=$(grep -n '^| `lane_branch.py`' training/README.md | head -1 | cut -d: -f1)
[ -n "$m" ] && sed -i "${m}s# |\$# Option \`use_depth\` / \`ModelConfig.lane_use_depth\` (default True; False = lift P: the same projection table without the depth weight, t24; the same tensors, \`tests/test_lane_pull.py\`). |#" training/README.md && echo "training README row extended at line $m"
# tests/README: working tree
k=$(grep -n '^| `test_lane_branch_dual.py`' tests/README.md | head -1 | cut -d: -f1)
[ -n "$k" ] && sed -i "${k}r $O/row_tests.txt" tests/README.md && echo "tests README row after line $k (working tree)"
# tests/README: blob = HEAD + the row
git show HEAD:bev_fusion/tests/README.md > $O/tests_readme_blob.md
k2=$(grep -n '^| `test_lane_branch_dual.py`' $O/tests_readme_blob.md | head -1 | cut -d: -f1)
sed -i "${k2}r $O/row_tests.txt" $O/tests_readme_blob.md && echo "blob built (row after line $k2)"
git diff --stat -- training/README.md training/configs/README.md | cat
cd /mnt/c/Auto-research && python3 autoresearch/tools/commit_paths.py --help 2>&1 | head -30
