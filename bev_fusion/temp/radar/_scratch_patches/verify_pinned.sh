#!/usr/bin/env bash
# runs tests/test_radar_slot.py on exactly the code that will be committed: HEAD's training/ + the three patched files, autogt v5 (Git Bash side
# builds the tree, WSL side runs it). Usage from the repository root: bash bev_fusion/temp/radar/_scratch_patches/verify_pinned.sh
set -e
cd "$(git rev-parse --show-toplevel)"
S=bev_fusion/temp/radar/_scratch_patches
V=$S/verify
rm -rf $V; mkdir -p $V
git archive HEAD bev_fusion/training | tar -x -C $V
git archive autogt/v5 bev_fusion/autogt | tar -x -C $V
for f in model train interface; do cp $S/headcopy/training/$f.py $V/bev_fusion/training/$f.py; done
mkdir -p $V/bev_fusion/tests && cp bev_fusion/tests/test_radar_slot.py $V/bev_fusion/tests/
mkdir -p $V/bev_fusion/data && ln -sfn "$(pwd)/bev_fusion/data/cache" $V/bev_fusion/data/cache 2>/dev/null || true
echo "tree ready: $V/bev_fusion"
