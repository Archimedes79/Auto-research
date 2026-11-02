#!/usr/bin/env bash
# the relative links to the moved plan documents (part 2 of the tidy), own lines only: tests/README.md as HEAD + my one path (others have uncommitted rows in it)
# run from the repository root: wsl bash bev_fusion/temp/pitch/commit_tidy2.sh
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -- bev_fusion/plan/README.md bev_fusion/experiments/t7_lane_branch/experiment.md bev_fusion/literature/code/fusion_blocks_comparison.md bev_fusion/training/legacy/README.md
git show "$old:bev_fusion/tests/README.md" | sed 's#(`../requirements.md`)#(`../plan/requirements.md`)#' > "$GIT_INDEX_FILE.readme"
blob=$(git hash-object -w "$GIT_INDEX_FILE.readme")
git update-index --cacheinfo "100644,$blob,bev_fusion/tests/README.md"
tree=$(git write-tree)
c=$(git commit-tree "$tree" -p "$old" -F bev_fusion/temp/pitch/commit_msg_tidy2.txt)
unset GIT_INDEX_FILE
git update-ref -m "bev_fusion tidy part 2" refs/heads/main "$c" "$old"
git reset -q HEAD -- bev_fusion/plan/README.md bev_fusion/experiments/t7_lane_branch/experiment.md bev_fusion/literature/code/fusion_blocks_comparison.md bev_fusion/training/legacy/README.md bev_fusion/tests/README.md
git log --oneline -1
git show --stat --format= "$c" | tail -7
