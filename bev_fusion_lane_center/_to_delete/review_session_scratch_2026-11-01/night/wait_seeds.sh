#!/bin/bash
for i in $(seq 1 115); do pgrep -f "[d]iag_seeds.py" >/dev/null || break; sleep 5; done
pgrep -f "[d]iag_seeds.py" >/dev/null && echo still running || echo done
