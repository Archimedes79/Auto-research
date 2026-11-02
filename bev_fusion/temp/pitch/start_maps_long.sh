#!/usr/bin/env bash
# the like-for-like map scoring of the 6000-step run under run.py, chained behind its evaluation (--after waits for the done flag, then for a GPU slot and < 3000 MB in use).
# Run it in the foreground (no nohup, the WSL session must stay alive).
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id pitch.maps_lfl_long --after t10.dual_long_s0.eval --gpu --gpu-max-mb 3000 --split val \
  --hyp "after four times the samples the dual chain stands higher against GaussianCaR on the same 400 frames than at 3000 steps (dual s0 / s1 shares: walkway 0.57 / 0.62, carpark 0.34 / 0.31, crossing 0.26 / 0.31, lane divider 0.50 / 0.52, road divider 0.72 / 0.76, drivable 0.94 / 0.94)" \
  --pred "against the mean of the two 3000-step shares: walkway, carpark, crossing and lane divider rise by 0.00 - 0.12 share points, drivable and road divider by 0.00 - 0.04, drivable stays below 0.98; no share falls more than 0.02 below that mean" \
  --rule "measurement, no decision of its own: the long run's shares stand next to the 3000-step shares wherever the case speaks of the long run; a share below the 3000-step mean by more than 0.03 (the effect of the threshold choice on the other layers) is reported as a loss, not rounded away" \
  -- bash temp/pitch/maps_gpu_long.sh > runs/pitch_maps_lfl_long.launcher.log 2>&1
echo "launcher exit $?"
