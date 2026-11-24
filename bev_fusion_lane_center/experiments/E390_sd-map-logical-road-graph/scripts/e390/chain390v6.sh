#!/bin/bash
# E390 step 1 chain (CPU only): per seed the arms g and n in parallel (2 threads each), then the VAL reads and the graph clause.
# usage: bash temp/e390/chain390.sh   (from the problem folder, in WSL)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e390/chain390v6.log
say() { echo "$(date '+%m-%d %H:%M:%S') [chain390] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
HYP="E390 step 1 (card v6 = v5 + the two P4 fixes + the P1 copy path, D130 / D135): a relation-biased graph encoding of the SD road graph (logical lanes per direction, links, junctions, route; node geometry cut at node boundaries) recovers lanes per direction, dropped links, hidden route flags and the association of perception proxies to their segment (with unassigned) above the measured nulls, invariant under rigid motion, with graceful decay"
PRED="(card v6, a second look at the same VAL frames) P4 passes; P1 at count error 0 about 0.99; P1 base about 0.94, P2 about 0.985, P3 about 0.85, P6 about 0.955 at base / 0.93 at pose2, none recall at nd about 0.8; SOUND 40 %, PARTLY 55 %"
RULE="E390 card v6 (section 4) with the unchanged rule and bars of section 0 (by hand, bars from the v6 nulls measured before the v5 training): SOUND = on both seeds P1 >= 0.7335 (base) and >= 0.99 (ce0), P2 >= 0.9380 / 0.9354 (base / s2), P3 >= 0.7138, P6 >= 0.9420 (base), >= 0.8747 (pose2), nd accuracy >= 0.8444 and none recall >= 0.3918, false-unassigned <= 0.05, each with the paired interval vs its null above 0; graph clause g - n on P3 >= +0.05; P4 <= 1e-4; P5 graceful; v6: P1 at count error 0 below 0.99 = a failing probe, the kill only below 0.90; order NOT SOUND > INCONCLUSIVE > PARTLY > SOUND; VAL 873, CPU, no TEST"
exec 8>/tmp/e390v6_chain.lock
flock -n 8 || { say "already runs"; exit 0; }
say "starts"
for S in 0 1; do
  for A in g6 n6; do
    ( $RUN --id E390.${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 180 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py train --arm $A --seed $S" >> temp/e390/${A}_s$S.out 2>&1; say "${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
done
for S in 0 1; do
  for A in g6 n6; do
    ( $RUN --id E390.read_${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py read --arm $A --seed $S" >> temp/e390/read_${A}_s$S.out 2>&1; say "read_${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
  $RUN --id E390.clause6_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py clause --seed $S --tag 6" >> temp/e390/clause6_s$S.out 2>&1
  say "clause6_s$S exit $?"
done
say "done"
