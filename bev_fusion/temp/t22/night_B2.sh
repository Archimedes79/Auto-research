#!/bin/bash
# scratch: Mirko's word (2026-11-08): stage B2 of the t22 card first. Starts only
#   - inside the night window 22:00 - 07:00 (my gate is night-only, D57: the lane work goes first by day),
#   - when the BEV system's t23.lane_bce (chain4, agreed order: lane, then t23, then mine) is done or cancelled, or from 01:00 on (it had its chance),
#   - after a 20-step smoke of the same config has finished (otherwise B2 is not started),
#   - when window_gate.py finds the GPU free (no CUDA process, memory < 2500 MB, util < 20 %, RAM >= 8 GB) on two polls in a row, and the run (3.3 h) plus 10 min
#     fits into the window (latest start 03:30); window_gate.py writes the trainer's stop word 5 minutes before 07:00;
# then the evaluation of the finished run (diagnostic) in the same window. Nothing here touches a running job of another chat.
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
log() { echo "$(date +%F_%T) $*"; }
t23_released() {
  python3 ../autoresearch/tools/schedule.py show 2>/dev/null | grep -E '^(done|cancelled)' | grep -q 't23.lane_bce' && return 0
  h=$(date +%H%M); [ "$h" -ge 0100 ] && [ "$h" -lt 0700 ] && return 0
  return 1
}
log "armed: waiting for the night window and for t23.lane_bce (done / cancelled, or 01:00)"
last=""
while true; do
  if t23_released; then break; fi
  cur="t23 not released at $(date +%H:%M)"
  [ "$cur" != "$last" ] && [ "$(date +%M)" -lt 5 ] && { log "$cur"; last="$cur"; }
  sleep 120
done
log "t23 released; 20-step smoke of B2 first (end-to-end proof, diagnostic), through window_gate"
python3 temp/t22/window_gate.py --night-only --hours 0.25 -- bash temp/t22/start_smoke_B2full.sh
if [ ! -f runs/smoke_t22_B2_s0/.done ]; then
  log "the smoke did not finish (no .done in runs/smoke_t22_B2_s0): B2 is NOT started; read runs/smoke_t22_B2_s0/train.log and records/runs/smoke.t22_B2_s0/run.json"
  exit 4
fi
log "smoke ok; handing over to window_gate for B2 (3.3 h)"
python3 temp/t22/window_gate.py --night-only --hours 3.3 --stop-file runs/t22_B2_s0/CONTROL -- bash temp/t22/start_B2.sh
rc=$?
log "window_gate for B2 returned $rc"
if [ -f runs/t22_B2_s0/.done ]; then
  log "t22_B2_s0 finished (.done): evaluation next"
  python3 temp/t22/window_gate.py --night-only --hours 0.4 -- bash temp/t22/start_evalB2.sh
  log "evaluation returned $?"
else
  log "no .done in runs/t22_B2_s0: no evaluation (stopped at the window end, failed or not started); state: records/runs/t22.B2_s0/run.json"
fi
log "night_B2 finished"
