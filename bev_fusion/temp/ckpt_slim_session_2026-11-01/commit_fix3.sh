set -e
cd /c/Auto-research
D=/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01
F=bev_fusion/training/train.py
VT=$(cat $D/fix3_tree.txt); V_OLD=$(cat $D/fix3_old.txt); OLD=$(git rev-parse HEAD)
echo "HEAD now $OLD, verified on top of $V_OLD"
[ "$(git rev-parse $OLD:$F)" = "$(git rev-parse $V_OLD:$F)" ] || { echo "$F changed in HEAD since the verified base: stop"; exit 1; }
export GIT_INDEX_FILE=$D/idx_fix3c
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
git update-index --cacheinfo 100644,$(git rev-parse $VT:$F),$F
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$D/commit_msg7.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- $F
rm -f $D/idx_fix3c
git show --stat --format='%h %s' HEAD | cut -c1-130
echo "working train.py differs from HEAD in: $(git diff HEAD --numstat -- $F | wc -l) file(s)"
