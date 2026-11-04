set -e
cd /c/Auto-research
SP=/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/8cc57966-d85d-48b0-a883-59cc642b2f8b/scratchpad
VERIFIED_TREE=$(cat "$SP/fix_tree.txt")
OLD=$(git rev-parse HEAD)
V_OLD=$(cat "$SP/fix_old.txt")
echo "HEAD now $OLD, verified on top of $V_OLD"
for f in bev_fusion/training/train.py bev_fusion/training/scratch.py; do
    a=$(git rev-parse "$OLD:$f"); b=$(git rev-parse "$V_OLD:$f")
    [ "$a" = "$b" ] || { echo "$f changed in HEAD since the verified base: stop"; exit 1; }
done
TRAIN_BLOB=$(git rev-parse "$VERIFIED_TREE:bev_fusion/training/train.py")
SCR_BLOB=$(git rev-parse "$VERIFIED_TREE:bev_fusion/training/scratch.py")
cmp -s <(git cat-file -p "$SCR_BLOB") bev_fusion/training/scratch.py || { echo "working scratch.py differs from the verified blob: stop"; exit 1; }
export GIT_INDEX_FILE=$SP/idx_fix2
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
git update-index --cacheinfo 100644,$TRAIN_BLOB,bev_fusion/training/train.py
git update-index --cacheinfo 100644,$SCR_BLOB,bev_fusion/training/scratch.py
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$SP/commit_msg4.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- bev_fusion/training/train.py bev_fusion/training/scratch.py
rm -f "$SP/idx_fix2"
git show --stat --format='%h %s' HEAD | cut -c1-150
echo "--- slim lines in the committed train.py by function:"
git show HEAD:bev_fusion/training/train.py | awk '/^def /{fn=$2} /slim/{print fn": "substr($0,1,90)}' | head -20
echo "--- train.py still differing from HEAD (others' hunks only?):"
git diff HEAD -- bev_fusion/training/train.py | grep -c '^[+-].*slim'
