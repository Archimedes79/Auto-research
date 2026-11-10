#!/usr/bin/env bash
# One-off (t3 report): qualitative images of the same 8 val frames for four systems, CPU only (the GPU belongs to the training lane).
# Separate run folders (copy of config.json) so the studied runs' own eval files are never touched.
cd "$(dirname "$0")/../../.." || exit 1
PY=~/miniconda3/envs/gcar/bin/python
D=/mnt/d/Auto-research/bev_fusion_ckpts
for pair in "t3_img_ref:t2_ref_s0:$D/t2_ref_s0" "t3_img_cam:t2_radar_off:$D/t2_radar_off" "t3_img_rad:t3_radar_only:$D/t3_radar_only" "t3_img_long:t1_r4_soa:runs/t1_r4_soa"; do
  IFS=: read name src ck <<< "$pair"
  mkdir -p runs/$name
  $PY - "$name" "$src" <<'PYEOF'
import json, sys
name, src = sys.argv[1:3]
c = json.load(open(f"runs/{src}/config.json"))
s = json.load(open("data/scenes_parts1_6.json"))
c["name"] = name
c["val_scenes"], c["train_scenes"] = s["val"], s["train"]
json.dump(c, open(f"runs/{name}/config.json", "w"), indent=1)
PYEOF
  echo "== $name ($src) $(date +%T)"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 nice -n 10 $PY -u evaluation/bev_suite.py runs/$name --workers 0 --device cpu --ckpt-dir $ck eval --n 40 --images 8 2>&1 | tail -n 1
done
echo "images finished $(date +%T)"
