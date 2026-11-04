#!/usr/bin/env bash
# ledger notes after the SECOND stop of t10.dual_long_s0 (single-quoted strings: no re-parse)
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/ledger.py note t10.dual_long_s0 \
  --result 'stopped twice on request of the user, no verdict: at step 4773 (14:14, trainsuite stop, exit 75) and again at step 5030 of 6000 (14:38, exit 75) after another session had restarted it at 14:20 (run.py --no-git with the pinned snapshot commit d14597e2e8, so the code was the original one; not started by this chat, the starter is unknown); the status failed is the launcher reading of exit 75, not a crash; full checkpoints ckpt_00005030.pt and ckpt_00005000.pt in runs/t10_dual_long_s0; attempts: 1 (09:01 - 12:09 CUDA unknown error at step 3200), 2 (12:11 - 14:14, resumed at 3000), 3 (14:20 - 14:38, resumed at 4773)' \
  --decision 'paused until the user says resume: the next start must be the same id on the pinned snapshot (the 14:20 start used run.py --no-git and the commit as the second argument of temp/train/run_ar.sh); the chained evaluation and the like-for-like map scoring are blocked by the stops and are started again behind the resumed training'
python3 ../autoresearch/tools/ledger.py note t10.dual_long_s0.eval \
  --result 'blocked twice: its predecessor t10.dual_long_s0 was stopped on request at step 4773 (14:14) and at step 5030 (14:38), exit 75' \
  --decision 'start again with temp/t10/start_long_eval.sh behind the resumed training; the prediction field of this row is the older 3b one, the valid prediction is the one of the training row'
python3 ../autoresearch/tools/ledger.py tail 2>&1 | grep -E 't10.dual_long|maps_lfl_long' | cut -c1-200
