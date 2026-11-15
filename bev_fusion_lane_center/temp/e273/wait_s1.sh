#!/bin/bash
# one-off: return when the E273 seed 1 launcher has ended (run record status no longer running) or after 3 h
cd "$(dirname "$0")/../.."
for i in $(seq 1 360); do
  if [ -f runs/e273_p_s1/.done ]; then echo "done flag"; break; fi
  if ! pgrep -f "qgraph_train.py --name e273_p_s1" > /dev/null; then echo "trainer process gone"; break; fi
  sleep 30
done
date
ls runs | grep e273_p_s1
tail -n 2 runs/e273_p_s1_p7/log.txt 2>/dev/null
