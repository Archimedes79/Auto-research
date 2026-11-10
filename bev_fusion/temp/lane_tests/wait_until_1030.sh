#!/bin/bash
# scratch: exits at Sunday 2026-11-08 10:30 (one notification to the BEV system chat: compile the final status before the 11:00 finish line)
target=$(date -d "2026-11-08 10:30" +%s)
while [ "$(date +%s)" -lt "$target" ]; do sleep 60; done
echo "10:30 reached: $(date)"
