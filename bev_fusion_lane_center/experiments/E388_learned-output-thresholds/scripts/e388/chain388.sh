#!/bin/bash
# E388 (card v2) chain, both seeds: (1) the labelled live TRAIN dump (GPU, run.py --gpu), (2) the fit (CPU, TRAIN only: the taus written before any VAL number),
# (3) the VAL read at the fitted cell paired with the hand cell on the same live dump (CPU). usage: bash temp/e388/chain388.sh   (from the problem folder, in WSL)
cd "$(dirname "$0")/../.."
export BFLC_SIG=1.0
L=temp/e388/chain388.log
say() { echo "$(date '+%m-%d %H:%M:%S') [chain388] $*" >> $L; }
source temp/e388/texts388.sh
RUN="python3 ../autoresearch/tools/run.py"
PY="env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 bash tools/env/_py14.sh"; PYR="bash tools/env/_pyr.sh"
exec 8>/tmp/e388_chain.lock
flock -n 8 || { say "already runs"; exit 0; }
say "starts"
H=$(date +%H%M); if [ "$H" -ge 0500 ] && [ "$H" -lt 2200 ]; then say "after the 05:00 start limit: nothing started"; exit 1; fi
for S in 0 1; do
  R=e372_s4_s$S
  if [ ! -f runs/$R/token_train_every3_live_labels.pkl ]; then
    $RUN --id E388.lab_s$S --seed $S --gpu --retries 1 --require-card --split val --timeout 45 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PYR jthresh.py dump --run $R --seed $S --split train --every 3" >> temp/e388/lab_s$S.out 2>&1
    say "lab_s$S exit $?"
  fi
done
for S in 0 1; do
  R=e372_s4_s$S
  $RUN --id E388.fit_s$S --seed $S --retries 0 --require-card --split val --timeout 30 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jthresh.py fit --run $R --seed $S --split train --every 3" >> temp/e388/fit_s$S.out 2>&1
  say "fit_s$S exit $?"
  CELL=$(python3 -c "import json; print(json.load(open('runs/$R/thresh_fit_train_every3.json'))['cell'])") || { say "no fit json for s$S"; continue; }
  IFS=, read LT LTL KT G <<< "$CELL"
  say "s$S cell from the TRAIN fit: $CELL (written before any VAL score)"
  $RUN --id E388.t_s$S --seed $S --retries 0 --require-card --split val --timeout 60 --hyp "$HYP" --pred "$PRED" --rule "$RULE" -- bash -c "$PY jtoken_eval.py score --live --run $R --seed $S --link-thr $LT --link-thr-long $LTL --keep-thr $KT --no-none-gate --ref-cell 0.3,0.2,0.6,off" >> temp/e388/t_s$S.out 2>&1
  say "t_s$S exit $?"
done
say "done"
