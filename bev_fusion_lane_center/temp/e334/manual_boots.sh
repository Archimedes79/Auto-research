#!/bin/bash
cd "$(dirname "$0")/../.."
for S in 0 1; do
  bash tools/env/_py14.sh jdiag.py boot e334_o_s$S ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 > temp/e334/boot_vs_e312_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e334_o_s$S ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 > temp/e334/boot_vs_e312_snap_s$S.log 2>&1
done
echo done > temp/e334/manual_boots.done
