#!/bin/bash
for i in $(seq 1 110); do pgrep -f "[d]iag_far.py" >/dev/null || break; sleep 5; done
pgrep -f "[d]iag_far.py" >/dev/null && echo still running || echo done
