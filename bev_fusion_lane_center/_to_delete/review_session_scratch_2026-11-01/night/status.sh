# night status: lane CPU arms, bev GPU queue, memory
cd /mnt/c/Auto-research/bev_fusion_lane_center
date +%T; uptime | sed 's/.*load/load/'; free -g | sed -n 2p; nvidia-smi --query-gpu=memory.used --format=csv,noheader
for f in temp/loop22/*.launcher.log; do echo "== $f"; tail -n 2 "$f" | cut -c1-200; done
pgrep -fa "run.py --id" | grep -v selftest | cut -c1-110
echo "bev t10_dual_s1: $(tail -n 1 ../bev_fusion/runs/t10_dual_s1/train.log | cut -c1-60)"
tail -n 2 /mnt/c/Users/mmeut/AppData/Local/Temp/claude/C--Auto-research/6e9638e0-a42f-4660-b7d5-cac910d9084d/tasks/bt7v94xsf.output
