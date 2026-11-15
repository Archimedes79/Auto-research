#!/bin/bash
# E351 quiet-host eager timing of E312 / E341 / E324 s / E348 (the lane lead's order of 2026-11-15 12:00: "the adopted removal's ms saving was not resolved on the loaded host"):
# waits (up to 14 h) for a quiet window = no run.py job with --gpu, GPU utilisation <= 5 %, 1-minute load < 3, three checks in a row one minute apart, then runs the E342 part B protocol
# (`jtiming.py run`: alternating blocks, paired differences with a bootstrap over the rounds, the section pass) through run.py. jtiming itself refuses a busy GPU (no --force).
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
ok=0; n=0
while [ $n -lt 840 ]; do
  gpu_jobs=$(pgrep -fc "run\.py --id .* --gpu" )
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  load=$(cut -d' ' -f1 /proc/loadavg)
  if [ "${gpu_jobs:-0}" -eq 0 ] && [ "${util:-100}" -le 5 ] && awk "BEGIN{exit !($load < 3.0)}"; then ok=$((ok+1)); else ok=0; fi
  if [ $ok -ge 3 ]; then break; fi
  n=$((n+1)); sleep 60
done
echo "quiet window after $n min: gpu_jobs=$gpu_jobs util=$util load=$load $(date +%H:%M:%S)"
[ $ok -ge 3 ] || { echo "no quiet window within 14 h"; exit 2; }
python3 ../autoresearch/tools/run.py --id E351.q_eager --seed 0 --retries 0 --split val --gpu --timeout 90 \
  --hyp "E351 / E342 part B on a QUIET host: the eager forward of E312, E341, E324 s and E348 in the same session (the loaded-host numbers of E342 part B were inflated by host contention): does removing the 0.5 m paint branch (-1.5 GMACs) buy about 1 ms?" \
  --pred "E341 minus E312 about -1 ms (E342 part B: -1.1 ms on a loaded host), E348 minus E324 s about -1 ms; E324 s minus E312 about +5.0 ms" \
  --rule "by hand: measurement, paired differences with a bootstrap over the rounds (jtiming.py run, the E342 part B protocol); quiet = load < 3 and the GPU idle before and after" \
  -- bash tools/env/_py14.sh jtiming.py run --runs e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0 --tag e351_quiet_eager
echo "q_eager exit $? $(date +%H:%M:%S)"
