set -e
cd /c/Auto-research
SP=/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/8cc57966-d85d-48b0-a883-59cc642b2f8b/scratchpad
OLD=$(git rev-parse HEAD)
git show "$OLD:bev_fusion/training/train.py" | cmp - "$SP/h_train.py" || { echo "HEAD train.py changed since it was corrected: stop"; exit 1; }
export GIT_INDEX_FILE=$SP/idx_fix
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
BLOB=$(git hash-object -w "$SP/r_train.py")
git update-index --cacheinfo 100644,$BLOB,bev_fusion/training/train.py
git add -- bev_fusion/training/scratch.py
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
echo "$OLD" > "$SP/fix_old.txt"
echo "$TREE" > "$SP/fix_tree.txt"
rm -rf "$SP/verify"
mkdir -p "$SP/verify"
git archive "$TREE" bev_fusion | tar -x -C "$SP/verify"
V=$SP/verify/bev_fusion
rm -rf "$V/data" "$V/models"
ln -s /c/Auto-research/bev_fusion/data "$V/data" 2>/dev/null || true
ln -s /c/Auto-research/bev_fusion/models "$V/models" 2>/dev/null || true
ls "$V" | tr '\n' ' '
echo
echo "tree $TREE"
