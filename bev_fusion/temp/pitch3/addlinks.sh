cd "$(git rev-parse --show-toplevel)"; REPO=$(pwd); EXP=$REPO/temp/accept_$(git rev-parse --short HEAD)
for f in bevfusion imagenet dualbev-r50-cbgs.pth; do ln -sfn "$REPO/bev_fusion/models/pretrained/$f" "$EXP/bev_fusion/models/pretrained/$f"; done
ls -la "$EXP/bev_fusion/models/pretrained" | cut -c1-140
rm -rf "$EXP"/bev_fusion/temp/tmp*          # the failed door copies
