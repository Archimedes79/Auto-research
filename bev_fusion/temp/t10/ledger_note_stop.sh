#!/usr/bin/env bash
# ledger notes after the stop of t10.dual_long_s0 (single-quoted strings: no re-parse)
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/ledger.py note t10.dual_long_s0 \
  --result 'stopped on request of the user at step 4773 of 6000 (trainsuite stop, checkpoint saved, exit 75; the status failed is the launcher reading of exit 75, not a crash); full checkpoints ckpt_00004773.pt and ckpt_00004500.pt in runs/t10_dual_long_s0; no cold-VAL number, no verdict; attempt 2 (resumed at step 3000 after the CUDA unknown error of 12:09) ran 12:11 - 14:17' \
  --decision 'paused: resume later with temp/t10/start_long.sh (same id; a new snapshot of the working tree, so the code drift against d14597e2e8 is checked first and written into the card); the chained evaluation and the like-for-like map scoring are blocked by this stop and are started again behind the resumed training'
python3 ../autoresearch/tools/ledger.py note t10.dual_long_s0.eval \
  --result 'blocked: its predecessor t10.dual_long_s0 was stopped on request at step 4773 (exit 75)' \
  --decision 'start again with temp/t10/start_long_eval.sh behind the resumed training; the prediction field of this row is the older 3b one, the valid prediction is the one of the training row'
python3 ../autoresearch/tools/ledger.py note pitch.maps_lfl_long \
  --result 'blocked: the evaluation it waits for is blocked by the stop of t10.dual_long_s0' \
  --decision 'start again with temp/pitch/start_maps_long.sh behind the resumed evaluation'
python3 ../autoresearch/tools/ledger.py tail 2>&1 | grep -E 't10.dual_long|maps_lfl_long' | cut -c1-200
cd ../autoresearch/tools && python3 machine_lock.py release bev-system 2>&1 | tail -2
