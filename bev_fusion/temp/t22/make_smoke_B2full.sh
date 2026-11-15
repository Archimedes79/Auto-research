#!/bin/bash
# scratch: write the 20-step smoke config of the resolved stage-B2 config (CPU only); the preview file is what the dry run of trainsuite start composed
cd /mnt/c/Auto-research/bev_fusion || exit 1
PY=$HOME/miniconda3/envs/gcar/bin/python
mkdir -p temp/t22/smoke
test -f temp/compose/t22_B2_s0.preview.json || { echo "no preview: run temp/t22/dry_B2.sh first"; exit 1; }
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 $PY training/startable.py temp/compose/t22_B2_s0.preview.json --write-smoke temp/t22/smoke 2>&1 | tail -6 | cut -c1-300
ls temp/t22/smoke
$PY - <<'EOF'
import json
c = json.load(open("temp/t22/smoke/smoke_t22_B2_s0.json"))
print(c["name"], "steps", c["steps"], "student_frames", c["model"].get("student_frames"), "camera_student", c["model"].get("camera_student"))
EOF
