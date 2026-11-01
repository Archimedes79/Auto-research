#!/usr/bin/env bash
# commits HEAD + the radar-slot wiring (patched HEAD copies, test, configs, doc rows) through a temporary index; the shared index and the
# working tree (other chats' uncommitted edits) are not touched. Usage from the repository root.
set -e
cd "$(git rev-parse --show-toplevel)"
S=bev_fusion/temp/radar/_scratch_patches
H=$S/headcopy
base=$(cat $S/base_commit.txt)
export GIT_INDEX_FILE="${TEMP:-/tmp}/idx_r2_slot"
rm -f "$GIT_INDEX_FILE"
git read-tree "$base"
put() {  # put <file> <repo path> [mode]
  local mode=${3:-100644}
  git update-index --add --cacheinfo "$mode,$(git hash-object -w "$1"),$2"
}
put $H/training/model.py bev_fusion/training/model.py
put $H/training/train.py bev_fusion/training/train.py
put $H/training/interface.py bev_fusion/training/interface.py
put $H/tests_README.md bev_fusion/tests/README.md
put $H/training_README.md bev_fusion/training/README.md
put $H/configs_README.md bev_fusion/training/configs/README.md
mode_gates=$(git ls-tree "$base" bev_fusion/tests/run_gates.sh | cut -d' ' -f1)
put $H/run_gates.sh bev_fusion/tests/run_gates.sh "$mode_gates"
put bev_fusion/tests/test_radar_slot.py bev_fusion/tests/test_radar_slot.py
for c in bev_fusion/training/configs/r2_rad_*.json; do put "$c" "$c"; done
git diff --cached --stat | tail -15
tree=$(git write-tree)
new=$(git commit-tree "$tree" -p "$base" -m "bev_fusion radar slot delta_slots: RadarEncoder as radar_kind in model.py, the reconstruction side loss (TrainConfig.radar_aux, decaying to a floor) and the kept-fraction monitor in train.forward_loss, interface doc, test_radar_slot.py (8 checks), six r2 comparison configs

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>")
git update-ref refs/heads/main "$new" "$base"
echo "committed $new"
unset GIT_INDEX_FILE
echo "$new" > $S/pinned_commit.txt
