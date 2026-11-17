#!/bin/bash
# E351 GPU smoke: CUDA-graph capture and the dump path on 8 frames (run.py --gpu, no card, no verdict)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
python3 ../autoresearch/tools/run.py --id E351.smoke_g1 --no-card --seed 0 --retries 0 --split val --gpu --timeout 20 \
  --hyp "E351 smoke: CUDA-graph capture and the dump path on 8 frames" \
  --pred "PARTIAL (8 of 873 frames), null and graph identical, 176 tensors (the CPU stand-in: 22 tensors per frame)" \
  --rule "by hand: smoke, no verdict" \
  -- bash tools/env/_py14.sh jgraph.py check --run e312_p_s0 --max 8 --dump --runs-dir temp/_session_scratch_e351/smoke_gpu
echo "smoke exit $?"
