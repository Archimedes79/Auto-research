#!/usr/bin/env bash
# applies the radar-slot wiring patches to the HEAD copies in headcopy/ (WSL side); run from the bev_fusion folder
set -e
cd "$(dirname "$0")/../../.."
P=~/miniconda3/envs/gcar/bin/python
S=temp/radar/_scratch_patches
cd $S/headcopy
for n in 1 2 3; do $P ../patch_wire$n.py; done
sed -i 's/{"radar_kind": "pillars"}, {"radar_kind": "off"}, {"trunk_kind"/{"radar_kind": "pillars"}, {"radar_kind": "delta_slots", "c_rad": 64}, {"radar_kind": "off"}, {"trunk_kind"/' training/model.py
cd ../../../..
$P $S/docs_head.py
grep -c "delta_slots" $S/headcopy/training/model.py
