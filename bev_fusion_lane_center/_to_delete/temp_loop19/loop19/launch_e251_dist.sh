#!/bin/bash
# D10 (a) (2026-10-25, Mirko: "fix data"): the b123 exact-distance map rebuilt exactly (CPU k-d tree) into a NEW file; the old file stays, nothing switches.
#   bash temp/loop19/launch_e251_dist.sh        -> ~/exp_gcar_b123x/feats/{train,val}_gt_{dist,pts}_exact.npy, rows in the CRN token order of exp_crn_b123
cd "$(dirname "$0")/../.."
mkdir -p "$HOME/exp_gcar_b123x/feats"
python3 ../autoresearch/tools/run.py --id E251.dist --retries 1 --timeout 240 \
  --hyp "data fix D10: the b123 exact-distance map rebuilt with the CPU k-d tree equals the float64 truth on every frame, and gt_pts_exact is unchanged" \
  --pred "10,120 rows (8,414 + 1,706); new map never above the 0.5 m-point bound by more than 0.01 px; gt_pts_exact bit-identical to the stored file; the stored map differs by > 0.05 px in about 40-45 % of frames (E235.chk3: 52 / 119 train, 26 / 119 val)" \
  --rule "PASS -> the new file is the exact target for the exact-vs-stored variant experiment; FAIL -> find the difference, the stored file stays the only target" \
  -- bash -c "GT_FRAME=ego NUSC_ROOT=\$HOME/data/nuscenes_full NUSC_VERSION=v1.0-trainval bash tools/env/_py14.sh make_dist_gt.py \$HOME/exp_gcar_b123x/feats \$HOME/exp_crn_b123/feats --cpu && bash tools/env/_py14.sh temp/loop19/check_b123x_dist.py"
