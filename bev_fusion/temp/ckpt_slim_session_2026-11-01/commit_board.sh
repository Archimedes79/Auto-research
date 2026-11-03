set -e
cd /c/Auto-research
SP=/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/8cc57966-d85d-48b0-a883-59cc642b2f8b/scratchpad
wsl -e bash -lc "cd /mnt/c/Auto-research/bev_fusion && ~/miniconda3/envs/gcar/bin/python evaluation/board.py --selftest | tail -1 && ~/miniconda3/envs/gcar/bin/python evaluation/board.py --write | tail -1"
if [ -n "$(git diff --cached --name-only)" ]; then echo "the shared index is not empty:"; git diff --cached --name-only; exit 1; fi
OLD=$(git rev-parse HEAD)
export GIT_INDEX_FILE=$SP/idx_board
rm -f "$GIT_INDEX_FILE"
git read-tree HEAD
git add -- bev_fusion/evaluation/board.py bev_fusion/records/champion.json bev_fusion/records/BOARD.md bev_fusion/records/README.md
git show HEAD:bev_fusion/evaluation/README.md > "$SP/readme_head.md"
row=$(grep -m1 '^| `board.py` |' bev_fusion/evaluation/README.md)
awk -v row="$row" '/^\| `bev_selftest.py` \|/ && !d {print row; d=1} {print}' "$SP/readme_head.md" > "$SP/readme_new.md"
sha=$(git hash-object -w "$SP/readme_new.md")
git update-index --cacheinfo 100644,$sha,bev_fusion/evaluation/README.md
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$SP/commit_msg.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- bev_fusion/evaluation/board.py bev_fusion/records/champion.json bev_fusion/records/BOARD.md bev_fusion/records/README.md bev_fusion/evaluation/README.md
rm -f "$SP/idx_board" "$SP/readme_head.md" "$SP/readme_new.md"
git show --stat --format='%h %s' HEAD | cut -c1-200
echo "--- still uncommitted in these folders:"
git status --short bev_fusion/evaluation bev_fusion/records
