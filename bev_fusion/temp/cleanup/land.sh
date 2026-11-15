#!/bin/bash
# land the integration branch (2 commits) into the main checkout as ONE commit built from the working tree + blobs for the files that carry foreign hunks
# usage: bash land.sh [--dry]
M=/c/Auto-research
W=$M/.claude/worktrees/integration
S=$M/bev_fusion/temp/cleanup
DRY=$1
cd $W || exit 1
base=$(git rev-parse integration~2); tip=$(git rev-parse main)
[ "$base" = "$tip" ] || { echo "main moved ($tip) since the integration base ($base): rebase integration first"; exit 1; }
git diff --no-renames --name-status integration~2 integration > $S/landing_status.txt
SKIP_BLOB="
.gitignore
bev_fusion/temp/README.md
bev_fusion/tests/README.md"
SKIP_ALL="$SKIP_BLOB
bev_fusion/temp/study/move_new_feats.py
bev_fusion/studies/recipes/study/move_new_feats.py"
cd $M && git status --porcelain | grep -v '^??' | cut -c4- | sort > $S/main_dirty_now.txt
awk '{print $2}' $S/landing_status.txt | sort > $S/landing_files_now.txt
bad=$(comm -12 $S/landing_files_now.txt $S/main_dirty_now.txt | grep -vxF "$SKIP_ALL")
[ -z "$bad" ] || { echo "ABORT: files of the landing carry uncommitted changes in main:"; echo "$bad"; exit 1; }
cd $W
paths=""; nd=0; nm=0
while read -r st p; do
  if echo "$SKIP_ALL" | grep -qx "$p"; then continue; fi
  if [ "$st" = "D" ]; then
    [ -e "$M/$p" ] && { [ "$DRY" = "--dry" ] || rm -f "$M/$p"; nd=$((nd+1)); }
  else
    [ "$DRY" = "--dry" ] || { mkdir -p "$(dirname "$M/$p")"; cp "$W/$p" "$M/$p"; }
    nm=$((nm+1))
  fi
  paths="$paths $p"
done < $S/landing_status.txt
echo "copied/added/modified: $nm, deleted: $nd, paths: $(echo $paths | wc -w)"
blobs=""
for p in $(echo "$SKIP_BLOB"); do
  b=$S/blob_$(echo $p | tr '/' '_')
  git show integration:$p > $b
  blobs="$blobs --blob $p=bev_fusion/temp/cleanup/blob_$(echo $p | tr '/' '_')"
done
echo "$paths" | tr ' ' '\n' | sed '/^$/d' > $S/landing_paths.txt
echo "blobs: $blobs"
