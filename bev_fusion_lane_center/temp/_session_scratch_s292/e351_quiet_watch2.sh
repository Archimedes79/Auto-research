#!/bin/bash
# E351 quiet-host eager timing, second try (the first try at 17:03 was refused by jtiming: GPU util max 6 % from desktop processes, no job of ours): poll nvidia-smi every 3 s until 12 consecutive samples show <= 3 % utilisation,
# the load is < 3 and no run.py --gpu job runs; then the same run.py job as before (same id, frozen hyp / pred / rule). Gives up at 18:25 (the Fable session needs both slots from 18:30).
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
ok=0
while [ "$(date +%H%M)" -lt 1825 ]; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  load=$(cut -d' ' -f1 /proc/loadavg)
  jobs=$(pgrep -fc "run\.py --id .* --gpu")
  if [ "${util:-100}" -le 3 ] && [ "${jobs:-0}" -eq 0 ] && awk "BEGIN{exit !($load < 3.0)}"; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 12 ] && break
  sleep 3
done
echo "quiet GPU after the poll: ok=$ok util=$util load=$load jobs=$jobs $(date +%H:%M:%S)"
[ $ok -ge 12 ] || { echo "no idle GPU before 18:25"; exit 2; }
python3 ../autoresearch/tools/run.py --id E351.q_eager --seed 0 --retries 0 --split val --gpu --timeout 90 \
  --hyp "E351 / E342 part B on a QUIET host: the eager forward of E312, E341, E324 s and E348 in the same session (the loaded-host numbers of E342 part B were inflated by host contention): does removing the 0.5 m paint branch (-1.5 GMACs) buy about 1 ms?" \
  --pred "E341 minus E312 about -1 ms (E342 part B: -1.1 ms on a loaded host), E348 minus E324 s about -1 ms; E324 s minus E312 about +5.0 ms" \
  --rule "by hand: measurement, paired differences with a bootstrap over the rounds (jtiming.py run, the E342 part B protocol); quiet = load < 3 and the GPU idle before and after" \
  -- bash tools/env/_py14.sh jtiming.py run --runs e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0 --tag e351_quiet_eager
echo "q_eager exit $? $(date +%H:%M:%S)"
