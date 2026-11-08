#!/usr/bin/env bash
# two commits of the tidy of the bev_fusion root through a temporary index, own paths only (run from the repository root: wsl bash bev_fusion/temp/pitch/commit_tidy.sh)
# 1 the tidy: plan.md / radar_encoding.md / requirements.md -> plan/, plan_draft2.md out of the tree, README rows, project.json structure lists; experiments/README.md as HEAD + my one line (others have uncommitted rows in it)
# 2 the purge: the tracked files of bev_fusion/_to_delete that Mirko purged (deletions only) + its README
set -e
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
HERE=bev_fusion/temp/pitch
commit_with() {   # commit_with <message file> <ref message>   (the index is prepared by the caller in $GIT_INDEX_FILE, HEAD tree already read)
  tree=$(git write-tree)
  c=$(git commit-tree "$tree" -p "$old" -F "$1")
  unset GIT_INDEX_FILE
  git update-ref -m "$2" refs/heads/main "$c" "$old"
  git log --oneline -1
  git show --stat --format= "$c" | tail -14
}
# ---- 1
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git add -A -- bev_fusion/plan.md bev_fusion/plan_draft2.md bev_fusion/radar_encoding.md bev_fusion/requirements.md bev_fusion/plan
git add -- bev_fusion/README.md bev_fusion/project.json bev_fusion/literature/README.md
git show "$old:bev_fusion/experiments/README.md" | sed 's#`../plan.md` section 11#`../plan/plan.md` section 11#' > "$GIT_INDEX_FILE.readme"
blob=$(git hash-object -w "$GIT_INDEX_FILE.readme")
git update-index --cacheinfo "100644,$blob,bev_fusion/experiments/README.md"
commit_with "$HERE/commit_msg_tidy.txt" "bev_fusion root tidy"
# ---- 2
old=$(git rev-parse HEAD)
export GIT_INDEX_FILE="$(mktemp -u)"
git read-tree "$old"
git ls-files -d -- bev_fusion/_to_delete | xargs -r git rm --cached -q --
git add -- bev_fusion/_to_delete/README.md
commit_with "$HERE/commit_msg_purge.txt" "bev_fusion _to_delete purge"
# refresh the shared index entries of the paths touched
git reset -q HEAD -- bev_fusion/plan.md bev_fusion/plan_draft2.md bev_fusion/radar_encoding.md bev_fusion/requirements.md bev_fusion/plan bev_fusion/README.md bev_fusion/project.json bev_fusion/literature/README.md bev_fusion/experiments/README.md bev_fusion/_to_delete 2>/dev/null || true
git status --short -- bev_fusion/plan bev_fusion/README.md bev_fusion/project.json bev_fusion/_to_delete bev_fusion/experiments/README.md | head -6
