#!/bin/bash
# one phase-timing run (the command of a run.py launch): phase_run.sh <tag> [<init checkpoint>]. Prints load, GPU memory, the sampler's tables and the GPU utilisation
# (nvidia-smi dmon, 1 s) inside the sampled window.
cd /mnt/c/Auto-research/bev_fusion || exit 1
tag=$1; init=${2:-}
out=temp/speed_tools/phase_$tag
echo "load before: $(cat /proc/loadavg)   GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)   free RAM: $(free -m | awk '/Mem:/{print $7" MB available"}')"
nvidia-smi dmon -s u -o T -d 1 > $out.dmon 2>&1 &
dm=$!
~/miniconda3/envs/gcar/bin/python -u temp/speed_tools/phase_sampler.py runs/t20_base_s0/code temp/compose/t20_base_s0.json "$tag" 44 10 $init 2>&1 | tee $out.txt
rc=${PIPESTATUS[0]}
kill $dm 2>/dev/null
echo "load after: $(cat /proc/loadavg)"
w=$(grep '^WINDOW' $out.txt | tail -1)
if [ -n "$w" ]; then
  set -- $w
  awk -v a="$2" -v b="$3" '$1 ~ /^[0-9][0-9]:/ && $1 >= a && $1 <= b {s += $3; n++} END {if (n) printf "GPU sm utilisation in the window: %.1f %% (%d one-second samples)\n", s / n, n}' $out.dmon
fi
echo "top 5 processes by CPU at the end:"; ps -eo pcpu,pmem,etime,args --sort=-pcpu | head -6 | cut -c1-140
exit $rc
