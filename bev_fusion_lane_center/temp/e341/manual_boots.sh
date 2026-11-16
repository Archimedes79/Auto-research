#!/bin/bash
# E341 paired scene bootstraps run by hand while the chain waits for GPU room (gpu_gate): the same commands as chain.sh, own logs (manual_boot_*.log).
cd "$(dirname "$0")/../.."
for S in 0 1; do
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e312_p_s$S ends_card_cnt_thr0.1 > temp/e341/manual_boot_vs_e312_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 > temp/e341/manual_boot_vs_e312_snap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_thr0.1 e332_pa_s$S ends_card_thr0.1 > temp/e341/manual_boot_vs_a_unsnap_s$S.log 2>&1
  bash tools/env/_py14.sh jdiag.py boot e341_p_s$S ends_card_snap_thr0.1 e332_pa_s$S ends_card_snap_thr0.1 > temp/e341/manual_boot_vs_a_snap_s$S.log 2>&1
done
echo "manual boots done $(date)" > temp/e341/manual_boots.done
