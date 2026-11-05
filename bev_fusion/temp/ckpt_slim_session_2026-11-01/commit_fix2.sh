set -e
cd /c/Auto-research
D=/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01
VT=$(cat $D/fix2_tree.txt); V_OLD=$(cat $D/fix2_old.txt); OLD=$(git rev-parse HEAD)
echo "HEAD now $OLD, verified on top of $V_OLD"
FILES="bev_fusion/training/train.py bev_fusion/training/scratch.py bev_fusion/training/README.md bev_fusion/temp/README.md"
for f in $FILES; do
    [ "$(git rev-parse $OLD:$f)" = "$(git rev-parse $V_OLD:$f)" ] || { echo "$f changed in HEAD since the verified base: stop"; exit 1; }
done
export GIT_INDEX_FILE=$D/idx_fix3
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
for f in $FILES; do
    git update-index --cacheinfo 100644,$(git rev-parse $VT:$f),$f
done
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$D/commit_msg5.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- $FILES
rm -f $D/idx_fix3
git show --stat --format='%h %s' HEAD | cut -c1-140
echo "--- default in the committed train.py:"
git show HEAD:bev_fusion/training/train.py | grep -n 'slim_at_end: bool' | cut -c1-60
echo "--- slim_at_end / keep lines of mine still differing from HEAD in the working tree (should be 0):"
git diff HEAD -- bev_fusion/training/train.py | grep -c '^[+-].*slim_at_end'
