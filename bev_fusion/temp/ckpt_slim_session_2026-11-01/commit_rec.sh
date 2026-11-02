set -e
cd /c/Auto-research
D=/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01
F=bev_fusion/records/CKPT_SLIM_2026-11-01.md
echo "diff of the record against HEAD (added/removed lines):"
git diff HEAD -U0 -- $F | grep '^[+-][^+-]' | cut -c1-110
OLD=$(git rev-parse HEAD)
export GIT_INDEX_FILE=$D/idx_rec
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
git add -- $F
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$D/commit_msg6.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- $F
rm -f $D/idx_rec
git show --stat --format='%h %s' HEAD | cut -c1-130
git status --short $F | head -2
