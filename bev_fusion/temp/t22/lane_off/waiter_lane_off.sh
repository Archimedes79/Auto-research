#!/bin/bash
# scratch: the L4 check (start_lane_off.sh) after my B2 chain has ended (rearm_B2_after_chain7.sh gone), inside the night window, GPU free on two polls (window_gate.py).
# Nothing here touches a job of another chat. The name matches neither t22/night_B2.sh (chain7's pattern) nor anything chain7's gpu_free pgrep looks for.
cd /mnt/c/Auto-research/bev_fusion || exit 1
log() { echo "$(date +%F_%T) $*"; }
log "armed: waiting for rearm_B2_after_chain7.sh to end, then window_gate (night only, 0.6 h) -> temp/t22/lane_off/start_lane_off.sh"
# also not while the BEV system's chain7 is still running (no DONE7 yet): if my B2 waiter had died, this one must not slip into the gap between two of their runs
while pgrep -f "t22/rearm_B2_after_chain7\.sh" > /dev/null || { [ ! -f temp/lane_tests/DONE7 ] && pgrep -f "lane_tests/chain7\.sh" > /dev/null; }; do sleep 120; done
log "the B2 chain has ended; handing over to window_gate"
python3 temp/t22/window_gate.py --night-only --hours 0.6 -- bash temp/t22/lane_off/start_lane_off.sh
log "window_gate returned $?"
for r in diag_t20_L4on diag_t20_L4off; do [ -f runs/$r/eval/card.json ] && log "$r: eval/card.json exists" || log "$r: NO eval/card.json"; done
log "waiter_lane_off finished"
