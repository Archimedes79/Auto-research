#!/bin/bash
# acceptance of "stored runs still work": a CLEAN git archive export of HEAD (no leftover dir, no link of autogt), a copy of two stored runs, read-only links to the data
set -e
cd "$(git rev-parse --show-toplevel)"
REPO=$(pwd); H=$(git rev-parse --short HEAD)
EXP=$REPO/temp/accept_$H
rm -rf "$EXP"; mkdir -p "$EXP"
git archive HEAD | tar -x -C "$EXP"
echo "export of $H in $EXP"
rm -rf "$EXP/data" "$EXP/bev_fusion/data"
ln -s "$REPO/data" "$EXP/data"
ln -s "$REPO/bev_fusion/data" "$EXP/bev_fusion/data"
mkdir -p "$EXP/bev_fusion/runs"
for r in t20_base_s0 t16_full_s0; do cp -r "$REPO/bev_fusion/runs/$r" "$EXP/bev_fusion/runs/$r"; done
echo "autogt in the export:"; ls "$EXP/autogt"
echo "GtGenerator in the export: $(ls $EXP/GtGenerator | wc -l) entries"
du -sh "$EXP" | tail -1
# the pretrained weights are not in git (ignored): read-only links to the real ones
for f in bevfusion imagenet dualbev-r50-cbgs.pth; do ln -sfn "$REPO/bev_fusion/models/pretrained/$f" "$EXP/bev_fusion/models/pretrained/$f"; done
