#!/usr/bin/env bash
# starts the six radar-slot comparison runs (experiment r2_radar_slot) as two chains (one per seed, arms one after the other, so at most 2 GPU
# slots are used): slots64 -> raster64 -> raster32. Code pinned to the commit in pinned_commit.txt, GT generator pinned to autogt v5 (the
# signature 15efbde2d1 of the t10 runs; their GT cache exists). Run from the bev_fusion folder in the WSL.
set -e
cd "$(dirname "$0")/../../.."
S=temp/radar/_scratch_patches
PIN=$(cat $S/pinned_commit.txt)
HYP='The delta-coded slot encoder (radar_kind delta_slots, c_rad 64, reconstruction side loss radar_aux 1.0 decaying to 0.05) lifts a radar-only BEV system above the width-matched raster block (c_rad 64) and the shipped one (c_rad 32), because it keeps the detections (unit test: 93 % kept with the side loss against 52 % with the task alone)'
PRED='radar-only, cold VAL 1000, slots64 minus raster64, paired bootstrap: vehicle seg iou@0 +0.02, front mAP2 +0.02, vehicle velocity skill (ave_skill_moving) +0.05, drivable iou@0.5 within +-0.01; radar_kept at the last validation >= 0.8'
RULE='per seed pair (s0 vs s0, s1 vs s1) the paired scene bootstrap of bev_suite compare: KEEP (build delta_slots into the default radar slot, propose the dual-chain check) if at least two of {vehicle iou@0, front mAP2, velocity skill} improve with the interval excluding 0 on BOTH seeds and drivable iou@0.5 is not worse by more than 0.01; TIE (keep raster, simpler) if inside the interval; CUT if two or more are worse with the interval excluding 0. raster64 - raster32 shows the effect of width alone'
launch() {  # launch <arm> <seed> <after-id or empty>
  local arm=$1 seed=$2 after=$3 name="r2_rad_$1_s$2" id="r2.$1_s$2"
  mkdir -p runs/$name
  if [ ! -d runs/$name/autogt ]; then git -C .. archive autogt/v5 bev_fusion/autogt | tar -x -C runs/$name --strip-components=1; fi
  local after_arg=""; [ -n "$after" ] && after_arg="--after $after"
  AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id $id --seed $seed --basis t3_radar_only --gpu --no-git $after_arg \
    --timeout 300 --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
    -- bash temp/train/run_ar.sh training/configs/$name.json $PIN > runs/$name.launcher.log 2>&1 &
  echo "launched $id (after: ${after:-none})"
}
for seed in 0 1; do
  launch slots64 $seed ""
  launch raster64 $seed "r2.slots64_s$seed"
  launch raster32 $seed "r2.raster64_s$seed"
done
