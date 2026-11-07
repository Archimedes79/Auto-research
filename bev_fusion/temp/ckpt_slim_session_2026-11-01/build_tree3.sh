set -e
cd /c/Auto-research
D=/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01
F=bev_fusion/training/train.py
echo "hunks of the working train.py against HEAD (expect 3, all mine):"
git diff HEAD -U0 --no-color -- $F | grep '^@@' | cut -c1-100
OLD=$(git rev-parse HEAD)
echo "$OLD" > $D/fix3_old.txt
cp bev_fusion/training/train.py $D/new3_train.py
export GIT_INDEX_FILE=$D/idx_fix3b
rm -f "$GIT_INDEX_FILE"
git read-tree "$OLD"
B=$(git hash-object -w "$D/new3_train.py")
git update-index --cacheinfo 100644,$B,$F
git diff --cached --numstat "$OLD"
TREE=$(git write-tree)
unset GIT_INDEX_FILE
echo "$TREE" > $D/fix3_tree.txt
rm -rf $D/verify3
mkdir -p $D/verify3
git archive "$TREE" bev_fusion/training bev_fusion/tests bev_fusion/evaluation bev_fusion/project.json autogt autoresearch | tar -x -C $D/verify3
wsl -e bash -lc "cd /mnt/c/Auto-research/bev_fusion/temp/ckpt_slim_session_2026-11-01/verify3/bev_fusion && ln -s /mnt/c/Auto-research/bev_fusion/data data && ln -s /mnt/c/Auto-research/bev_fusion/models models && ~/miniconda3/envs/gcar/bin/python -m py_compile training/train.py && echo compiled"
echo "tree $TREE"
