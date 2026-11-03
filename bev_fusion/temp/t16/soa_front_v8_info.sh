cd /mnt/c/Auto-research/bev_fusion
D=~/data/nuscenes_full/v1.0-trainval
head -c 900 $D/sample_data.json; echo; echo ----
head -c 700 $D/sample.json; echo; echo ----
head -c 500 $D/calibrated_sensor.json; echo; echo ----
head -c 400 $D/scene.json; echo
head -c 300 runs/t16_full_s0/eval/suite/quant/eval_curves_fusion_vel_n1000/manifest.json; echo
grep -c . runs/t16_full_s0/eval/suite/quant/eval_curves_fusion_vel_n1000/frames.jsonl
