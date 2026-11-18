#!/usr/bin/env bash
# Loop 3, E38/E39: Vorschlaege aus den dichten Koepfen (sobald jeder Lauf fertig ist), Kaskaden-Verfeinerer (Haupt, danach Ablationen), Verbindungskopf (run_stageB.sh).
S=/mnt/c/ML-Experiment/bev_fusion_lane_center/training
O=/mnt/c/ML-Experiment/bev_fusion_lane_center/runs
G="/mnt/c/ML-Experiment/bev_fusion_lane_center/temp/archive/misc/run_gcar.sh"
waitfor() { until [ -f "$1" ]; do sleep 15; done; }
for k in 0 1 2; do
  ( waitfor $O/e37_fold$k/metrics.json; echo "== $(date +%H:%M:%S) Vorschlaege fold$k"; $G proposals.py --name e37_fold$k --set oof --tag relax --workers 4 2>&1 | tail -1; $G proposals.py --name e37_fold$k --set oof --tag relax --spacing 2.0 --workers 4 2>&1 | tail -1 ) &
done
( waitfor $O/e37_final/metrics.json; echo "== $(date +%H:%M:%S) Vorschlaege final"
  for st in val test train_eval; do $G proposals.py --name e37_final --set $st --tag relax --ceiling --workers 4 2>&1 | tail -2 | cut -c1-400; done
  for st in val test; do $G proposals.py --name e37_final --set $st --tag relax --spacing 2.0 --workers 4 2>&1 | tail -1; done ) &
wait
echo "== $(date +%H:%M:%S) Vorschlaege fertig -> Verfeinerer"
TR="e37_fold0:oof:train,e37_fold1:oof:train,e37_fold2:oof:train"
EV="val=e37_final:val:val,test=e37_final:test:val,oofA=e37_fold0:oof:train"
run() { name="$1"; shift; if [ -f "$O/$name/metrics.json" ]; then echo "uebersprungen: $name"; return; fi; echo "== $(date +%H:%M:%S) $name $*"; $G refiner.py --name "$name" --workers 4 "$@" 2>&1 | grep -E "Parameter|Trainings|\[val\]|\[test\]|\[oofA\]|FERTIG|Error|Traceback" | cut -c1-330; }
run e39_main --train $TR --eval $EV --steps 5000 --blocks 3 --lam_fp 3 --jitter 0.5
( export GPU_PRIO=low; run e39_b1 --train $TR --eval $EV --steps 5000 --blocks 1 --lam_fp 3 --jitter 0.5; run e39_nojit --train $TR --eval $EV --steps 5000 --blocks 3 --lam_fp 3 --jitter 0.0 ) &
( export GPU_PRIO=low; run e39_fp1 --train $TR --eval $EV --steps 5000 --blocks 3 --lam_fp 1 --jitter 0.5; run e39_s2 --tag relax_s2 --spacing 2.0 --train $TR --eval $EV --steps 5000 --blocks 3 --lam_fp 3 --jitter 0.5 ) &
wait
echo "== $(date +%H:%M:%S) E39 fertig"
