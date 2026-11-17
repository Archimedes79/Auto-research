#!/bin/bash
# the champion's whole-path timing in the quiet window (lane lead 03:15): waits until no other run.py job is active and the 1-min load is below 2.5, then times the
# E365 S1 head (the champion architecture; the decode costs nothing measurable) with B' seed 0 through run.py (GPU ticket); twice (repeat 2). usage: bash temp/e371/time.sh
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
source temp/e371/texts.sh
say() { echo "$(date '+%m-%d %H:%M:%S') [timing] $*" >> temp/e371/chain.log; }
say "waiting for the quiet window"
while pgrep -f "autoresearch/tools/run.py" > /dev/null || [ "$(awk '{print ($1 < 2.5)}' /proc/loadavg)" != 1 ]; do sleep 15; done
say "quiet: load $(cut -d' ' -f1 /proc/loadavg), no run.py job; timing starts"
for k in 1 2; do
  python3 ../autoresearch/tools/run.py --id E371.time$k --seed 0 --gpu --retries 0 --require-card --split val --timeout 20 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash tools/env/_py14.sh jtoken_eval.py timing --run e365_s1_s0 --b-run runs/e292_a_s0 >> temp/e371/time.out 2>&1
  say "timing $k exit $?"
done
say "window closed"
