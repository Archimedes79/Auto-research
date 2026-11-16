#!/bin/bash
cd "$(dirname "$0")/../.."
b() { echo "## $*" >> temp/e323/boot.log; nice -n 19 bash tools/env/_py14.sh jdiag.py boot "$@" 2>&1 | tail -n 6 >> temp/e323/boot.log; }
rm -f temp/e323/boot.log
for P in "0 0.083 0.12" "1 0.09 0.11"; do
  set -- $P; S=$1; T1=$2; T2=$3
  b e320_a_s$S ends_card_snap_nogeom_attach_thr0.1 e312_p_s$S ends_card_snap_thr$T1
  b e320_a_s$S ends_card_snap_nogeom_attach_thr0.1 e312_p_s$S ends_card_snap_nogeom_thr$T1
  b e312_p_s$S ends_card_snap_thr$T1 e312_p_s$S ends_card_snap_thr0.1
  b e320_a_s$S ends_card_snap_nogeom_attach_thr$T2 e312_p_s$S ends_card_snap_thr0.1
done
echo done > temp/e323/boot.done
