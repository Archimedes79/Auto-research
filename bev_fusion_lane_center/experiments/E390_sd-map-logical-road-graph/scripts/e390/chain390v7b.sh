#!/bin/bash
# E390 step 1 card v7 chain, CONTINUATION after the trainings (the first chain's holdout reads were refused by run.py: split train_holdout is not a project split; run.py now gets --split val, the read itself is on the TRAIN holdout scenes) (CPU only): per seed the arms g7 and n7 in parallel (2 threads each, the TRAIN holdout scenes left out), then the TRAIN-holdout reads (BEFORE VAL), then the VAL reads and the graph clause.
# usage: bash temp/e390/chain390v7.sh   (from the problem folder, in WSL)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e390/chain390v7.log
say() { echo "$(date '+%m-%d %H:%M:%S') [chain390v7] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
HYP="E390 step 1 (card v7 = v6 + the map's stated count-error rate as an input, D130 / D135): a relation-biased graph encoding of the SD road graph (logical lanes per direction, links, junctions, route; node geometry cut at node boundaries) recovers lanes per direction (also copying the exact input), dropped links, hidden route flags and the association of perception proxies to their segment (with unassigned) above the measured nulls, invariant under rigid motion, with graceful decay"
PRED="(card v7, the third look at the same VAL frames) P1 at count error 0 about 0.998, P1 base about 0.94 or higher, the other probes as in v6, P4 0.0; SOUND 70 %, PARTLY 25 %"
RULE="E390 card v7 (section 5) with the unchanged rule and bars of section 0 (by hand, bars from the v6 nulls measured before the v5 training): SOUND = on both seeds P1 >= 0.7335 (base) and >= 0.99 (ce0), P2 >= 0.9380 / 0.9354 (base / s2), P3 >= 0.7138, P6 >= 0.9420 (base), >= 0.8747 (pose2), nd accuracy >= 0.8444 and none recall >= 0.3918, false-unassigned <= 0.05, each with the paired interval vs its null above 0; graph clause g - n on P3 >= +0.05; P4 <= 1e-4; P5 graceful; P1 at count error 0 below 0.99 = a failing probe, the kill only below 0.90; order NOT SOUND > INCONCLUSIVE > PARTLY > SOUND; VAL 873, CPU, no TEST"
exec 8>/tmp/e390v7b_chain.lock
flock -n 8 || { say "already runs"; exit 0; }
say "continuation starts (holdout reads, stop rule, VAL)"
for S in 0 1; do                                     # the TRAIN-holdout reads, recorded BEFORE any VAL read
  for A in g7 n7; do
    ( $RUN --id E390.hold_${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py read --arm $A --seed $S --split holdout" >> temp/e390/hold_${A}_s$S.out 2>&1; say "hold_${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
done
# the stop rule (card v7, written before the run): P1 at count error 0 on the TRAIN holdout below 0.99 on either seed of g7 -> no VAL read
GO=$(python3 -c "
import json
v = [json.load(open('runs/e390_g7_s%d/probe_card_holdout.json' % s))['levels']['ce0']['p1'][0] for s in (0, 1)]
print('go' if min(v) >= 0.99 else 'stop', v)")
say "holdout stop rule: $GO"
case "$GO" in go*) ;; *) say "stop before VAL (card v7 stop rule)"; exit 0;; esac
for S in 0 1; do
  for A in g7 n7; do
    ( $RUN --id E390.read_${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py read --arm $A --seed $S" >> temp/e390/read_${A}_s$S.out 2>&1; say "read_${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
  $RUN --id E390.clause7_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py clause --seed $S --tag 7" >> temp/e390/clause7_s$S.out 2>&1
  say "clause7_s$S exit $?"
done
say "done"
