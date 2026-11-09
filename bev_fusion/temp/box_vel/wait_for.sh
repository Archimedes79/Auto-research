#!/usr/bin/env bash
# t11 vel cause: wait (inside the WSL) until a file exists or extract_all.log reports a failure / skip / the end, then print what happened.
# Run from bev_fusion/: bash temp/box_vel/wait_for.sh temp/box_vel/out/t8_ctl_s0_val.npz
cd "$(dirname "$0")/../.." || exit 1
log=temp/box_vel/out/extract_all.log
until [ -f "$1" ] || grep -q -E "FAILED|SKIPPED|Traceback|all done" "$log"; do sleep 20; done
grep -E "wrote|FAILED|SKIPPED|Traceback|Error|all done" "$log" | tail -5
