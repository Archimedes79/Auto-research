#!/bin/bash
# scratch: re-arm of the B2 night chain after the reboot of 2026-11-08 22:09 (night_B2.sh died with WSL; B2 and its smoke never started).
# Order agreed with the BEV system chat: its chain7 (t23 second seed: t21.ctl_s1 + eval, t23.C_s1 + eval, about 4.2 h from 22:16) goes first, mine after it.
# Released when temp/lane_tests/DONE7 exists (written by chain7 at its end), or when chain7 is gone for 3 polls in a row without writing it (crashed).
# Then the same steps as night_B2.sh: 20-step smoke (B2 is not started when it did not finish), B2 (3.3 h, latest start about 03:30 so that it ends before 07:00), evaluation.
# This file is NOT called night_B2.sh on purpose: chain7 waits for any process matching t22/night_B2.sh and would never start its second run.
# B2_FLAG / B2_DRY are for the self-test only (a dry run stops after the release).
cd /mnt/c/Auto-research/bev_fusion || exit 1
export AR_PROJECT=../bev_fusion
FLAG=${B2_FLAG:-temp/lane_tests/DONE7}
log() { echo "$(date +%F_%T) $*"; }
chain7_alive() { pgrep -f "lane_tests/chain7\.sh" > /dev/null; }
log "armed: waiting for $FLAG (end of the BEV system's chain7), then smoke, B2, evaluation through window_gate (night window, GPU free on two polls)"
gone=0; last=""
while true; do
  [ -f "$FLAG" ] && { log "$FLAG exists: chain7 is over"; break; }
  if chain7_alive; then gone=0; else gone=$((gone + 1)); fi
  [ "$gone" -ge 3 ] && { log "chain7 is not running and wrote no $FLAG (3 polls): treating it as over"; break; }
  cur="waiting for chain7 at $(date +%H):xx"
  [ "$cur" != "$last" ] && [ "$(date +%M)" -lt 5 ] && { log "$cur"; last="$cur"; }
  sleep 120
done
[ -n "$B2_DRY" ] && { log "dry run: stop here"; exit 0; }
log "released; 20-step smoke of B2 first (end-to-end proof, diagnostic), through window_gate"
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
log "rearm_B2_after_chain7 finished"
