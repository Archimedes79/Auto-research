set -e
cd /c/Auto-research
SP=/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/8cc57966-d85d-48b0-a883-59cc642b2f8b/scratchpad
[ -z "$(git diff --cached --name-only)" ] || { echo "the shared index holds staged files:"; git diff --cached --name-only; }
OLD=$(git rev-parse HEAD)
export GIT_INDEX_FILE=$SP/idx3
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
git add -- bev_fusion/records/CKPT_SLIM_2026-11-01.md bev_fusion/records/README.md
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$SP/commit_msg3.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- bev_fusion/records/CKPT_SLIM_2026-11-01.md bev_fusion/records/README.md
rm -f "$SP/idx3"
git show --stat --format='%h %s' HEAD | cut -c1-140
git status --short bev_fusion/records | head -5
