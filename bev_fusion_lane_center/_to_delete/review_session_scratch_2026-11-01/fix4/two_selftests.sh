cd /mnt/c/Auto-research/bev_fusion_lane_center
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for k in 1 2; do (nice python3 ../autoresearch/tools/selftest.py > /tmp/ar_par_$k.txt 2>&1; echo "run $k exit $?") & done
wait
for k in 1 2; do grep -B4 -A2 'Traceback' /tmp/ar_par_$k.txt | head -12; tail -1 /tmp/ar_par_$k.txt; done
uptime
