#!/bin/bash
# E292 post processing of one trained arm: the B' full prediction dump with the graph cards (GPU, about 5 minutes, one at a time because of the host RAM), then the snapped scoring with E312's connectors on the delivered graph of B'
# (c2: read out with the hard step close:1.5@0.5; a: plain@0.5, and for a also with the hard step on top = the STRUCTURAL check). Cards: runs/e312_p_s<S>/ends_card_<tag>_snap_thr0.1.json (tagged: never overwritten, the card of record c stays).  usage: post_arm.sh <c2|a> <seed>
cd "$(dirname "$0")/../.."
ARM=$1; S=$2; R=e292_${ARM}_s$S; L=temp/day15/day15.log
[ -f runs/$R/model.pt ] || { echo "no model of $R $(date)" >> $L; exit 1; }
mkdir -p temp/e292
flock /tmp/jends_dump.lock bash tools/env/_pyr.sh qgraph_run_eval.py score --run $R --seed $S --exp e292 --readouts plain@0.5 close:1.5@0.5 --dump-full >> temp/e292/dump_$R.log 2>&1
echo "dump $R exit $? $(date)" >> $L
FULL=$(pwd)/temp/e292/predfull_$R.pkl; BASE=$(pwd)/runs/$R/model.pt
case $ARM in
  c2) SPECS="close:1.5@0.5:c2";;
  a)  SPECS="plain@0.5:a close:1.5@0.5:ah";;
esac
for SP in $SPECS; do
  SPEC=${SP%:*}; TAG=${SP##*:}
  bash tools/env/_py14.sh jends_eval.py score --run e312_p_s$S --seed $S --thr 0.1 --snap --full $FULL --base $BASE --spec $SPEC --tag $TAG >> temp/e292/score_${TAG}_s$S.log 2>&1
  echo "scored $R as $TAG exit $? $(date)" >> $L
done
