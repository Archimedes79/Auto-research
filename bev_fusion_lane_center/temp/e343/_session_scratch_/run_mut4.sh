#!/bin/bash
cd /mnt/c/Auto-research/bev_fusion_lane_center
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ "$1" = "0" ]; then
  NAMES=("sidecar old head refused" "precheck no flagged/default check" "precheck default without sidecar ok" "message interrupted" "precheck ctl_b always named" "code dirs no subfolders" "code lazy never followed" "code lazy followed in autoresearch" "code scratch walked" "code skip evaluate off" "code digest no stamp size" "provenance line removed dump_b" "provenance line removed dump_ends" "get_dump no provenance" "get_dump flagged unchecked" "get_dump default no sidecar ok" "provenance missing sidecar ok" "code no parent packages")
else
  NAMES=("expect ignored" "expect short prefix ok" "expect digest not printed" "expect card names none" "atomic no allow-list" "writable too broad" "writable prefix no sep" "writable no casefold" "precheck seed unchecked flagged" "dump_b overwrites" "dump_ends overwrites" "is_protected no realpath" "is_protected no casefold" "is_protected roots no casefold" "cmd_all no gate" "main dump no gate" "e321_changed empty" "tol REHEARSAL always" "check_stored ignores spec")
fi
nice -n 10 bash tools/env/_py14.sh temp/e343/_session_scratch_/mutate.py "${NAMES[@]}" > temp/e343/_session_scratch_/mutate_r3_$1.log 2>&1
