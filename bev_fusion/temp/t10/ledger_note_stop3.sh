#!/usr/bin/env bash
# correction of the ledger note of t10.dual_long_s0: the starter of the 14:20 attempt is known now (single-quoted strings: no re-parse)
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/ledger.py note t10.dual_long_s0 \
  --result 'stopped twice on request of the user, no verdict: at step 4773 (14:14, trainsuite stop, exit 75) and again at step 5030 of 6000 (14:38, exit 75) after the GT Discussion chat had restarted it at 14:20 on the order of the user (run.py --no-git with the pinned snapshot commit d14597e2e8, so the code was the original one; this chat did not know, the user was asked and answered stop it again); the status failed is the launcher reading of exit 75, not a crash; full checkpoints ckpt_00005030.pt and ckpt_00005000.pt in runs/t10_dual_long_s0; attempts: 1 (09:01 - 12:09 CUDA unknown error at step 3200), 2 (12:11 - 14:14, resumed at 3000), 3 (14:20 - 14:38, resumed at 4773)' \
  --decision 'paused until the user says resume (the user decided at about 14:30 to stop it again and later told the Architecture chat to build the target architecture first); the next start must be the same id on the pinned snapshot (run.py --no-git and the commit as the second argument of temp/train/run_ar.sh); the chained evaluation and the like-for-like map scoring are blocked by the stops and are started again behind the resumed training'
python3 ../autoresearch/tools/ledger.py tail 2>&1 | grep -E 't10.dual_long' | cut -c1-200
