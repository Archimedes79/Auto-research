#!/bin/bash
# Post-run waiter for E320 (2026-11-13 00:25, lane lead fork): day12c cannot be edited, so this separate waiter reads the pre-registered read-outs of the E320 card after day12c's own post-processing (E320_DONE).
# Per seed: PRIMARY (matched wiring) `--attach --snap --snap-nogeom`, the tight-wiring read-outs `--attach --attach-r 1.5 / 1.0`, the same run's keypoint ends with the matched wiring `--snap --snap-nogeom`, the maneuver variants, the maps score, the attached-end error (jdiag kperr) and the paired scene bootstrap. CPU only, nice. Log temp/day12/day12d.log. Never edit while it runs.
cd "$(dirname "$0")/../.."
L=temp/day12/day12d.log
echo "day12d waits for E320_DONE $(date)" >> $L
until [ -f temp/day12/E320_DONE ]; do sleep 60; done
echo "E320_DONE $(date)" >> $L
for S in 0 1; do
  R=e320_a_s$S
  [ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; continue; }
  nice bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --snap --snap-nogeom --thr 0.1 >> temp/day12/d_primary_$R.log 2>&1
  echo "primary scored $R $(date)" >> $L
  nice bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --snap --snap-nogeom --thr 0.1 >> temp/day12/d_kpwired_$R.log 2>&1
  nice bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --attach-r 1.5 --thr 0.1 >> temp/day12/d_tight15_$R.log 2>&1
  nice bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S --attach --attach-r 1.0 --thr 0.1 >> temp/day12/d_tight10_$R.log 2>&1
  echo "tight scored $R $(date)" >> $L
  nice bash tools/env/_py14.sh jmaneuver.py score --run $R --seed $S --variants plain p0.1 p0.1s p0.1an p0.1a p0.1ar1.5 p0.1ar1.0 >> temp/day12/d_maneuver_$R.log 2>&1
  nice bash tools/env/_pyr.sh jmaps_eval.py score --run ${R}_maps >> temp/day12/d_maps_$R.log 2>&1
  nice bash tools/env/_py14.sh jdiag.py kperr $R e312_p_s$S >> temp/day12/d_kperr_$R.log 2>&1
  nice bash tools/env/_py14.sh jdiag.py boot $R ends_card_snap_nogeom_attach_thr0.1 e312_p_s$S ends_card_snap_thr0.1 >> temp/day12/d_boot_$R.log 2>&1
  echo "scored $R $(date)" >> $L
done
touch temp/day12/DAY12D_DONE
echo "day12d done $(date)" >> $L
