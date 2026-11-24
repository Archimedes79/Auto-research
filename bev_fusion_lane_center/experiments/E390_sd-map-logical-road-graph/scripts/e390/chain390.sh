#!/bin/bash
# E390 step 1 chain (CPU only): per seed the arms g and n in parallel (2 threads each), then the VAL reads and the graph clause.
# usage: bash temp/e390/chain390.sh   (from the problem folder, in WSL)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e390/chain390.log
say() { echo "$(date '+%m-%d %H:%M:%S') [chain390] $*" >> $L; }
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"
HYP="E390 step 1 (card v5, D130 / D135): a relation-biased graph encoding of the SD road graph (logical lanes per direction, links, junctions, route; node geometry cut at node boundaries) recovers lanes per direction, dropped links, hidden route flags and the association of perception proxies to their segment (with unassigned) above the measured nulls, invariant under rigid motion, with graceful decay"
PRED="P1 base about 0.74 (copy 0.696), P2 about 0.94 (logistic 0.917), P3 about 0.80 (rule 0.618), P6 about 0.95 at base (nearest 0.923), 0.88 at pose2, none recall at nd about 0.5 (nearest 0.19); SOUND 30 %, PARTLY 50 %"
RULE="E390 card v5 section 0 (by hand, bars from the v6 nulls measured before training): SOUND = on both seeds P1 >= 0.7335 (base) and >= 0.99 (ce0), P2 >= 0.9380 / 0.9354 (base / s2), P3 >= 0.7138, P6 >= 0.9420 (base), >= 0.8747 (pose2), nd accuracy >= 0.8444 and none recall >= 0.3918, false-unassigned <= 0.05, each with the paired interval vs its null above 0; graph clause g - n on P3 >= +0.05; P4 <= 1e-4; P5 graceful; order NOT SOUND > INCONCLUSIVE > PARTLY > SOUND; VAL 873, CPU, no TEST"
exec 8>/tmp/e390_chain.lock
flock -n 8 || { say "already runs"; exit 0; }
say "starts"
for S in 0 1; do
  for A in g n; do
    ( $RUN --id E390.${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 180 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py train --arm $A --seed $S" >> temp/e390/${A}_s$S.out 2>&1; say "${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
done
for S in 0 1; do
  for A in g n; do
    ( $RUN --id E390.read_${A}_s$S --seed $S --retries 0 --require-card --split val --timeout 90 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py read --arm $A --seed $S" >> temp/e390/read_${A}_s$S.out 2>&1; say "read_${A}_s$S exit $?" ) &
    sleep 15
  done
  wait
  $RUN --id E390.clause_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY sdgraph_probe.py clause --seed $S" >> temp/e390/clause_s$S.out 2>&1
  say "clause_s$S exit $?"
done
say "done"
