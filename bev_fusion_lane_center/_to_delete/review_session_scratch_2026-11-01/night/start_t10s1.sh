cd /mnt/c/Auto-research/bev_fusion
S=/mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/scratchpad/night
python3 $S/make_cfg.py training/configs/t10_dual_s0.json training/configs/t10_dual_s1.json name='"t10_dual_s1"' seed=1
AR_PROJECT=../bev_fusion nohup python3 ../autoresearch/tools/run.py --id t10.dual_s1 --gpu --split train \
  --hyp "seed twin of t10.dual_s0 (its rule E1 held: maps of t8_bevfeat kept, front mAP2 0.33 >= ctl - 0.054)" \
  --pred "cold VAL 1000 within the t10 bands of s0: lane divider@0.5 0.34, drivable 0.80, walkway 0.33, front mAP2 0.33" \
  --rule "s1 agrees with s0 (E1 again) -> the dual-feature chain is the new reference (board base proposal; promotion stays Mirko's); s1 fails E1 -> t10 is seed luck, keep t8_bevfeat / t8_ctl" \
  -- bash temp/train/run_ar.sh training/configs/t10_dual_s1.json > runs/t10_dual_s1.launcher.log 2>&1 &
sleep 2; cat runs/t10_dual_s1.launcher.log
