set -e
cd /c/Auto-research
SP=/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/8cc57966-d85d-48b0-a883-59cc642b2f8b/scratchpad
OLD=$(git rev-parse HEAD)
export GIT_INDEX_FILE=$SP/idx_mine
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
git add -- bev_fusion/training/scratch.py bev_fusion/tests/test_flow_timed.py bev_fusion/tests/test_flow_fused.py bev_fusion/tests/test_flow_explains.py
part() {   # <path> <regex>: only the hunks whose added lines match
    git diff HEAD -U0 --no-color -- "$1" | awk -v pat="$2" -f "$SP/keep_hunks.awk" > "$SP/part.patch"
    [ -s "$SP/part.patch" ] || { echo "no hunk of $1 matches $2"; exit 1; }
    git apply --cached --unidiff-zero "$SP/part.patch"
    echo "$1: $(grep -c '^@@' "$SP/part.patch") hunk(s) of $(git diff HEAD -U0 --no-color -- "$1" | grep -c '^@@')"
}
part bev_fusion/training/train.py 'slim|.rng. in ck|.opt. in ck'
part bev_fusion/training/README.md 'scratch[.]py'
part bev_fusion/temp/README.md 't5 feature dumps|Size rules'
part bev_fusion/experiments/README.md 'scratch left behind'
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
C=$(git commit-tree "$TREE" -p "$OLD" -F "$SP/commit_msg2.txt")
git update-ref HEAD "$C" "$OLD"
git reset -q HEAD -- bev_fusion/training/scratch.py bev_fusion/tests/test_flow_timed.py bev_fusion/tests/test_flow_fused.py bev_fusion/tests/test_flow_explains.py bev_fusion/training/train.py bev_fusion/training/README.md bev_fusion/temp/README.md bev_fusion/experiments/README.md
rm -f "$SP/idx_mine" "$SP/part.patch"
echo "committed $C"
git show --stat --format='%h %s' HEAD | cut -c1-150
git show HEAD:bev_fusion/training/train.py > "$SP/train_committed.py"
