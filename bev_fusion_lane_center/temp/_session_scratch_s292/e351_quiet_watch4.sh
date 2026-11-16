#!/bin/bash
# The ONE quiet-host window of the night (D107, lane lead 2026-11-15 22:20): 05:00 - 07:00 after E357 / E353 / E354 are through. Fires once, in this order:
#   (1) E351.q_eager: eager forward of E312, E341, E324 s and E348 on a quiet host (jtiming run), frozen hyp / pred / rule of watch3;
#   (2) E353.time_s0: the speed half of E353 (jnarrow_eval.py time paired, E324 s against the narrowed n1), only if runs/e353_n1_s0/model.pt exists;
#   then the flag temp/_session_scratch_s292/quiet_window_done is written (its content says which parts ran): the lead of the Fable session is told by hand (its jrecur_time.py needs the same window next, ~10 min).
# Quiet = 20 consecutive nvidia-smi samples (3 s apart) with util <= 3 %, load < 3, no run.py --gpu job. No GPU job of ours starts in between. Gives up at 07:00 ("ms open" in the report).
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
FLAG=temp/_session_scratch_s292/quiet_window_done
rm -f $FLAG
START=$(date -d "tomorrow 05:00" +%s)
[ -n "$QUIET_NOW" ] && START=$(date +%s)
DEADLINE=$(date -d "tomorrow 07:00" +%s)
[ -n "$QUIET_NOW" ] && DEADLINE=$(( $(date +%s) + 7200 ))
while [ "$(date +%s)" -lt "$START" ]; do sleep 60; done
quiet_wait() { # quiet_wait: 0 when the host was quiet for 20 samples before the deadline
  local ok=0 util load jobs
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    load=$(cut -d' ' -f1 /proc/loadavg)
    jobs=$(pgrep -fc "run\.py --id .* --gpu")
    if [ "${util:-100}" -le 3 ] && [ "${jobs:-0}" -eq 0 ] && awk "BEGIN{exit !($load < 3.0)}"; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 20 ] && { echo "quiet: util=$util load=$load jobs=$jobs $(date +%H:%M:%S)"; return 0; }
    sleep 3
  done
  echo "no quiet window before the deadline: util=$util load=$load jobs=$jobs $(date +%H:%M:%S)"
  return 1
}
done_parts=""
if quiet_wait; then
  python3 ../autoresearch/tools/run.py --id E351.q_eager --seed 0 --retries 0 --split val --gpu --timeout 40 \
    --hyp "E351 / E342 part B on a QUIET host: the eager forward of E312, E341, E324 s and E348 in the same session (the loaded-host numbers of E342 part B were inflated by host contention): does removing the 0.5 m paint branch (-1.5 GMACs) buy about 1 ms?" \
    --pred "E341 minus E312 about -1 ms (E342 part B: -1.1 ms on a loaded host), E348 minus E324 s about -1 ms; E324 s minus E312 about +5.0 ms" \
    --rule "by hand: measurement, paired differences with a bootstrap over the rounds (jtiming.py run, the E342 part B protocol); quiet = load < 3 and the GPU idle before and after" \
    -- bash tools/env/_py14.sh jtiming.py run --runs e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0 --tag e351_quiet_eager
  echo "q_eager exit $? $(date +%H:%M:%S)"
  done_parts="$done_parts q_eager"
  if [ -f runs/e353_n1_s0/model.pt ]; then
    sleep 20
    if quiet_wait; then
      python3 ../autoresearch/tools/run.py --id E353.time_s0 --seed 0 --retries 0 --split val --gpu --timeout 25 \
        --hyp "E353 speed half on a QUIET host: the 0.5 m paint branch at 2 of 8 GroupNorm groups (n1) against the same recipe's E324 s (control), paired, one process (jnarrow_eval.py time paired)" \
        --pred "n1 minus E324 s about -0.8 ms (a guess from the E342 part B whole-branch -1.1 ms on a LOADED host: not transferable; the measurement decides)" \
        --rule "by hand: measurement, paired difference with a bootstrap over the rounds; quiet = both loads < 3 and the GPU at most 5 % before and after; D106 'faster' = the upper end of the interval is below 0 on a QUIET label" \
        -- bash tools/env/_py14.sh jnarrow_eval.py time paired --runs e324_s_s0,e353_n1_s0 --tag e353_time_s0
      echo "e353_time exit $? $(date +%H:%M:%S)"
      done_parts="$done_parts e353_time_s0"
    fi
  else
    echo "no runs/e353_n1_s0/model.pt: E353 speed half skipped"
  fi
fi
echo "parts run:${done_parts:- none} at $(date +%Y-%m-%d_%H:%M:%S)" > $FLAG
echo "window over: $(cat $FLAG)"
