#!/bin/bash
# clean stop of t10_dual_long_s0 at the user's request (trainsuite stop protocol: the trainer sees CONTROL at the next step boundary, saves a full checkpoint, writes STOPPED, exits 75)
cd "$(dirname "$0")/../../../autoresearch" || exit 1
python3 -m trainsuite --root ../bev_fusion/runs status t10_dual_long_s0 2>&1 | tail -6
python3 -m trainsuite --root ../bev_fusion/runs stop t10_dual_long_s0 --reason "stopped again at the user's request on 2026-11-01 after a restart by another session at 14:20; resume only on the user's word" 2>&1 | tail -12
