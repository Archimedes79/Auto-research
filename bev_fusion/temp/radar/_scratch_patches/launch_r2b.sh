#!/usr/bin/env bash
# restart of the radar-slot comparison after the RAM / swap exhaustion of 2026-11-01 17:2x (two runs with 6 workers each + other jobs: swap
# 16.2 of 16.4 GB, the runs stood at step 1): ONE chain, one run at a time, 2 loader workers, and every start waits for >= 9 GB of free RAM.
# Same pinned code (the commit in pinned_commit.txt) and GT (autogt v5); only `workers` differs from the committed configs (noted in _about).
# Order: the matched pair of seed 0 first (first read after about 2.5 h), then seed 1, then the shipped width.
set -e
cd "$(dirname "$0")/../../.."
S=temp/radar/_scratch_patches
PIN=$(cat $S/pinned_commit.txt)
mkdir -p temp/radar/r2_cfg
~/miniconda3/envs/gcar/bin/python - <<'PYEOF'
import glob, json, os
for p in sorted(glob.glob("training/configs/r2_rad_*.json")):
    c = json.load(open(p, encoding="utf-8"))
    c["workers"] = 2
    c["_about"] = "r2_radar_slot; the committed config (commit pinned in temp/radar/_scratch_patches/pinned_commit.txt) with workers 2 instead of 6: the first start exhausted RAM and swap on the shared box"
    json.dump(c, open("temp/radar/r2_cfg/" + os.path.basename(p), "w", encoding="utf-8"), indent=1)
print("configs ok")
PYEOF
HYP='The delta-coded slot encoder (radar_kind delta_slots, c_rad 64, reconstruction side loss radar_aux 1.0 decaying to 0.05) lifts a radar-only BEV system above the width-matched raster block (c_rad 64) and the shipped one (c_rad 32), because it keeps the detections (unit test: 93 % kept with the side loss against 52 % with the task alone)'
PRED='radar-only, cold VAL 1000, slots64 minus raster64, paired bootstrap: vehicle seg iou@0 +0.02, front mAP2 +0.02, vehicle velocity skill (ave_skill_moving) +0.05, drivable iou@0.5 within +-0.01; radar_kept at the last validation >= 0.8'
RULE='per seed pair (s0 vs s0, s1 vs s1) the paired scene bootstrap of bev_suite compare: KEEP (build delta_slots into the default radar slot, propose the dual-chain check) if at least two of {vehicle iou@0, front mAP2, velocity skill} improve with the interval excluding 0 on BOTH seeds and drivable iou@0.5 is not worse by more than 0.01; TIE (keep raster, simpler) if inside the interval; CUT if two or more are worse with the interval excluding 0. raster64 - raster32 shows the effect of width alone'
prev=""
for spec in slots64:0 raster64:0 slots64:1 raster64:1 raster32:0 raster32:1; do
  arm=${spec%%:*}; seed=${spec##*:}; name="r2_rad_${arm}_s${seed}"; id="r2.${arm}_s${seed}"
  after_arg=""; [ -n "$prev" ] && after_arg="--after $prev"
  AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id $id --seed $seed --basis t3_radar_only --gpu --no-git $after_arg \
    --timeout 400 --hyp "$HYP" --pred "$PRED" --rule "$RULE" \
    -- bash $S/wait_ram_then.sh bash temp/train/run_ar.sh temp/radar/r2_cfg/$name.json $PIN > runs/$name.launcher.log 2>&1 &
  echo "queued $id (after: ${prev:-none})"
  prev=$id
done
