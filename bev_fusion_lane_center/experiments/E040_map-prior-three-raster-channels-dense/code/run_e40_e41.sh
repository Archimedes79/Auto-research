#!/usr/bin/env bash
# Loop 3, E40 (Karten-Vorwissen am Gitter) und E41 (Zeit, Variante A am Gitter): Feinabstimmung des Endkopfes e37_final (2000 Schritte) mit/ohne Zusatz-Eingaben + Auswertungen mit anderen Stoerungen.
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
until [ -f $O/e37_final/metrics.json ]; do sleep 20; done
echo "== $(date +%H:%M:%S) e37_final fertig -> E40/E41"
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
FT="--init_from e37_final --steps 2000 --lr 5e-4 --bs 8 --workers 3 --eval train_eval,val,test"
EV="--steps 0 --eval val,test --workers 3"
run() { name="$1"; shift; if [ -f "$O/$name/metrics.json" ]; then echo "uebersprungen: $name"; return; fi; echo "== $(date +%H:%M:%S) $name $*"; $G dense_ext.py --name "$name" "$@" 2>&1 | grep -E "Eingangs|\[train_eval\]|\[val\]|\[test\]|FERTIG|Error|Traceback" | cut -c1-300; }
(
  run e40_nomap_ft $FT
  run e40_noise $FT --prior map --prior_xy 1.0 --prior_deg 2
  run e40_clean $FT --prior map
  run e40_oracle $FT --prior gt_oracle
  run e40_wrong $FT --prior map_wrong
  run e41_A $FT --temporal A --t_gap 1
) &
(
  until [ -f $O/e40_noise/metrics.json ]; do sleep 20; done
  for sig in 0 2 3; do
    run e40_noise_s${sig}_noalign $EV --init_from e40_noise --prior map --prior_xy 1.0 --prior_deg 2 --eval_prior_xy $sig --eval_prior_deg $((2*sig)) --eval_align 0
    run e40_noise_s${sig}_align   $EV --init_from e40_noise --prior map --prior_xy 1.0 --prior_deg 2 --eval_prior_xy $sig --eval_prior_deg $((2*sig)) --eval_align 1
  done
  run e40_noise_s1_align $EV --init_from e40_noise --prior map --prior_xy 1.0 --prior_deg 2 --eval_prior_xy 1 --eval_prior_deg 2 --eval_align 1
  run e40_noise_stale3 $EV --init_from e40_noise --prior map --prior_xy 1.0 --prior_deg 2 --eval_degrade 0.3
  run e40_noise_stale6 $EV --init_from e40_noise --prior map --prior_xy 1.0 --prior_deg 2 --eval_degrade 0.6
  until [ -f $O/e40_clean/metrics.json ]; do sleep 20; done
  run e40_clean_s1_noalign $EV --init_from e40_clean --prior map --eval_prior_xy 1.0 --eval_prior_deg 2 --eval_align 0
  run e40_clean_s1_align   $EV --init_from e40_clean --prior map --eval_prior_xy 1.0 --eval_prior_deg 2 --eval_align 1
) &
wait
echo "== $(date +%H:%M:%S) E40/E41 fertig"
