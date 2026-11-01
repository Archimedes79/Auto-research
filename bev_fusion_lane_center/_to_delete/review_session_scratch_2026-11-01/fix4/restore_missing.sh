# write back the files of bev_fusion_lane_center/temp that HEAD holds and the disk lacks; an existing file is never touched
cd /c/Auto-research
n=0; skip=0
while IFS= read -r p; do
  if [ -e "$p" ]; then skip=$((skip+1)); continue; fi
  mkdir -p "$(dirname "$p")"
  git show "HEAD:$p" > "$p" && n=$((n+1))
done < <(git ls-tree -r --name-only HEAD bev_fusion_lane_center/temp)
echo "restored $n, existing (left alone) $skip"
git status --short -- bev_fusion_lane_center/temp | head -5
