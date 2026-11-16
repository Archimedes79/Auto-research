#!/bin/bash
# E353: the paired scene bootstrap (jdiag.py boot, 1000 resamples of the VAL scenes) of n1 minus c (E324 s of the same seed) and n1 minus cp, per seed and judged read-out: the ADOPT-FIRM / POINT label of the card (lower bound of the topo F1 difference to c at least -0.010 on every seed and read-out).
# usage: e353_boots.sh <seed>   (CPU, nice 19, one process; log temp/e353/boot_<control>_<readout>_s<seed>.log)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
S=$1
mkdir -p temp/e353
for ro in snap primary attach; do
  case $ro in
    snap) card=ends_card_snap_thr0.1;;
    primary) card=ends_card_snap_nogeom_attach_thr0.1;;
    attach) card=ends_card_attach_thr0.1;;
  esac
  for ctl in e324_s_s$S:c e353_cp_s$S:cp; do
    run=${ctl%%:*}; tag=${ctl##*:}
    nice -n 19 bash tools/env/_py14.sh jdiag.py boot e353_n1_s$S $card $run $card > temp/e353/boot_${tag}_${ro}_s$S.log 2>&1
    echo "done $tag $ro s$S rc=$? $(date +%H:%M:%S)" >> temp/e353/boots.log
  done
done
echo "ALL s$S $(date +%H:%M:%S)" >> temp/e353/boots.log
