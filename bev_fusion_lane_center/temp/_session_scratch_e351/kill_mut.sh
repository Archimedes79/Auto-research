#!/bin/bash
# stop my own mutation batch (and only it)
for p in $(pgrep -f "[r]un_mut.sh"); do kill "$p"; done
for p in $(pgrep -f "_session_scratch_e351/mut/"); do kill "$p"; done
sleep 1
ps aux | grep -E "[r]un_mut|[s]cratch_e351/mut" | cut -c1-160
echo stopped
