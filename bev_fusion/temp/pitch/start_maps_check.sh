#!/usr/bin/env bash
# the layer-1 check of the GPU map scoring under run.py (D23): same model, GPU against the stored CPU numbers. Foreground (no nohup); waits for a GPU slot and < 3000 MB in use.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id pitch.maps_lfl_check --gpu --gpu-max-mb 3000 --split val \
  --hyp "the GPU path of the map scoring reproduces the stored CPU numbers of t1_r4_soa (400 frames), so the four like-for-like scorings of pitch.maps_lfl are as good as the CPU ones" \
  --pred "headline numbers of ours within 0.01 of the CPU run (float noise only), the GaussianCaR block identical" \
  --rule "a headline difference above 0.01 or a different GaussianCaR block: the GPU shares are not quoted until the cause is found; otherwise the GPU numbers stand" \
  -- bash temp/pitch/maps_gpu_check.sh > runs/pitch_maps_lfl_check.launcher.log 2>&1
echo "launcher exit $?"
