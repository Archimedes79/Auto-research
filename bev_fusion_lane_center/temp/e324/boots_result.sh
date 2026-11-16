#!/bin/bash
# E324 result bootstraps (scene resamples, jdiag.py boot): the same-read-out reference, the floor u', the adoption read-out (own keypoints snapped) against E312 snapped and E316 snapped. usage: boots_result.sh
cd "$(dirname "$0")/../.."
for S in 0 1; do
  nice bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_snap_nogeom_attach_thr0.1 e316_p_s$S ends_card_snap_nogeom_thr0.1 > temp/e324/boot_prim_vs_lnogeom_s$S.log 2>&1
  nice bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_snap_nogeom_attach_thr0.1 e312_p_s$S ends_card_snap_nogeom_thr0.1 > temp/e324/boot_prim_vs_u_s$S.log 2>&1
  nice bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_snap_thr0.1 e312_p_s$S ends_card_snap_thr0.1 > temp/e324/boot_ownsnap_vs_c_s$S.log 2>&1
  nice bash tools/env/_py14.sh jdiag.py boot e324_s_s$S ends_card_snap_thr0.1 e316_p_s$S ends_card_snap_thr0.1 > temp/e324/boot_ownsnap_vs_l_s$S.log 2>&1
done
echo done > temp/e324/boots_result.done
