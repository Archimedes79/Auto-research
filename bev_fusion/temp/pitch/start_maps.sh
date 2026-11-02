#!/usr/bin/env bash
# start the like-for-like map scoring of the case under run.py (D23): waits for a GPU slot and for the GPU memory in use to fall below 3000 MB.
# Start it ONLY when the queue owner's round has ended (the Architecture chat forwards that message); run it in the foreground (no nohup, the WSL session must stay alive).
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id pitch.maps_lfl --gpu --gpu-max-mb 3000 --split val \
  --hyp "on identical frames the dual chain stands at least at the conservative shares of the case, because the 400 GaussianCaR frames are easier for our models than our 1000 frames" \
  --pred "shares of GaussianCaR on the same 400 frames, dual chain: drivable 0.90-0.95, walkway 0.57-0.65, carpark 0.25-0.40, crossing 0.28-0.35, lane divider 0.48-0.56, road divider 0.70-0.80; control and BEVFusion-only at their ladder values" \
  --rule "measurement, no decision of its own: the shares replace the indicative table in the case (brief section 4, deck slide soa); a share more than 0.05 from the conservative estimate corrects the brief" \
  -- bash temp/pitch/maps_gpu.sh > runs/pitch_maps_lfl.launcher.log 2>&1
echo "launcher exit $?"
