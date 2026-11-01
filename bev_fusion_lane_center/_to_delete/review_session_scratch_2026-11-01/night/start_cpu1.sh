cd /mnt/c/Auto-research/bev_fusion_lane_center
nohup python3 ../autoresearch/tools/run.py --id E261.g --require-card --split val --basis E219 -- bash temp/loop22/run_E261g.sh > temp/loop22/E261g.launcher.log 2>&1 &
nohup python3 ../autoresearch/tools/run.py --id E265.c --require-card --split val --basis E259 -- bash temp/loop22/run_E265c.sh > temp/loop22/E265c.launcher.log 2>&1 &
sleep 20; tail -3 temp/loop22/E261g.launcher.log temp/loop22/E265c.launcher.log
