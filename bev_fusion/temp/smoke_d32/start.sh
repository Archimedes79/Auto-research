#!/usr/bin/env bash
# smoke test c of the D32 merge: a 20-step dual training (t11 hurdle + accumulation 2) through run.py, as the 6000-step run will start.
# Run in the foreground from anywhere (it changes to bev_fusion/); it waits for a GPU slot and for the GPU memory in use to fall below 3000 MB.
cd "$(dirname "$0")/../.." || exit 1
export AR_PROJECT=../bev_fusion
python3 ../autoresearch/tools/run.py --id smoke.d32 --gpu --gpu-max-mb 3000 --split train \
  --hyp "smoke test of the D32 merge: the launcher, the snapshot, the dual chain with the t11 hurdle and accumulation 2, one validation pass and a checkpoint write work end to end" \
  --pred "exit 0 within 15 min; a ledger row, tag bf/smoke.d32, run.json with the commit, runs/smoke_d32/.done and ckpt_00000020.pt" \
  --rule "any missing piece = the merge is not ready: fix before the 6000-step run; all present = the 6000-step run may start" \
  -- bash temp/train/run_ar.sh temp/smoke_d32/smoke_d32.json > runs/smoke_d32.launcher.log 2>&1
echo "launcher exit $?"
# step e: the evaluation, chained through run.py
python3 ../autoresearch/tools/run.py --id smoke.d32.eval --after smoke.d32 --gpu --gpu-max-mb 3000 --split val \
  --hyp "smoke test e of the D32 merge: the suite evaluates the smoke checkpoint and records the GT signature" \
  --pred "exit 0; eval/metrics + card with protocol.gt_signature 15efbde2d1 (the GT of the 6000-step run)" \
  --rule "missing card or another signature = fix before the 6000-step run" \
  -- bash temp/smoke_d32/eval.sh > runs/smoke_d32.eval.launcher.log 2>&1
echo "eval launcher exit $?"
