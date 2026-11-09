set -e
cd /c/Auto-research
D=/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01
OLD=$(git rev-parse HEAD)
echo "$OLD" > $D/fix2_old.txt
git show "$OLD:bev_fusion/training/train.py" > $D/head_train.py
git show "$OLD:bev_fusion/training/scratch.py" > $D/head_scratch.py
git show "$OLD:bev_fusion/training/README.md" > $D/head_training_README.md
git show "$OLD:bev_fusion/temp/README.md" > $D/head_temp_README.md
wsl -e bash -lc "cd /mnt/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01 && ~/miniconda3/envs/gcar/bin/python fix2.py /mnt/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01"
export GIT_INDEX_FILE=$D/idx_fix2
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
for pair in "train.py:bev_fusion/training/train.py" "scratch.py:bev_fusion/training/scratch.py" "training_README.md:bev_fusion/training/README.md" "temp_README.md:bev_fusion/temp/README.md"; do
    f=${pair%%:*}; p=${pair#*:}
    B=$(git hash-object -w "$D/new_$f")
    git update-index --cacheinfo 100644,$B,$p
done
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
echo "$TREE" > $D/fix2_tree.txt
rm -rf $D/verify2
mkdir -p $D/verify2
git archive "$TREE" bev_fusion/training bev_fusion/tests bev_fusion/evaluation bev_fusion/project.json autogt autoresearch | tar -x -C $D/verify2
V=$D/verify2/bev_fusion
wsl -e bash -lc "cd /mnt/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01/verify2/bev_fusion && ln -s /mnt/c/Auto-research/bev_fusion/data data && ln -s /mnt/c/Auto-research/bev_fusion/models models && ls"
echo "tree $TREE"
