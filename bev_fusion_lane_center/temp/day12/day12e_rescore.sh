#!/bin/bash
# Re-score of E312's snapped and snap-nogeom cards once (2026-11-13 00:40, lane lead fork; critic N3): runs/e312_p_s<S>/ends_card_snap_thr0.1.json and ends_card_snap_nogeom_thr0.1.json predate the .counts.pkl and the counts keys (long_links_dropped, within-1 m share), so `jdiag.py boot` cannot read them. Waits for DAY12D_DONE (day12d_post: the E320 read-outs), backs each card up (.pre_counts), re-scores, checks that the numbers are EQUAL to the backup (else restores it), then runs the paired scene bootstrap a - c and a - u' per seed. CPU, nice. Log temp/day12/day12e.log. Never edit while it runs.
cd "$(dirname "$0")/../.."
L=temp/day12/day12e.log
echo "day12e waits for DAY12D_DONE $(date)" >> $L
until [ -f temp/day12/DAY12D_DONE ]; do sleep 60; done
echo "DAY12D_DONE $(date)" >> $L
for S in 0 1; do
  R=e312_p_s$S
  for CARD in ends_card_snap_thr0.1 ends_card_snap_nogeom_thr0.1; do
    F=runs/$R/$CARD.json
    [ -f $F ] || { echo "missing $F $(date)" >> $L; continue; }
    cp $F $F.pre_counts
    if [ "$CARD" = ends_card_snap_thr0.1 ]; then FL="--snap"; else FL="--snap --snap-nogeom"; fi
    nice bash tools/env/_py14.sh jends_eval.py score --run $R --seed $S $FL --thr 0.1 >> temp/day12/e_rescore_${R}_$CARD.log 2>&1
    if python3 temp/day12/cmp_cards.py $F.pre_counts $F >> $L 2>&1; then echo "re-scored $R $CARD equal $(date)" >> $L; else cp $F.pre_counts $F; echo "RESTORED $F $(date)" >> $L; fi
  done
  nice bash tools/env/_py14.sh jdiag.py boot e320_a_s$S ends_card_snap_nogeom_attach_thr0.1 $R ends_card_snap_thr0.1 >> temp/day12/e_boot_c_$S.log 2>&1
  nice bash tools/env/_py14.sh jdiag.py boot e320_a_s$S ends_card_snap_nogeom_attach_thr0.1 $R ends_card_snap_nogeom_thr0.1 >> temp/day12/e_boot_u_$S.log 2>&1
  echo "boot done seed $S $(date)" >> $L
done
touch temp/day12/DAY12E_DONE
echo "day12e done $(date)" >> $L
