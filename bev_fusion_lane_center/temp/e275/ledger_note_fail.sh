#!/bin/bash
# one-off (E275): the judgement columns of the failed first attempt of E275.n_s0
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/ledger.py note E275.n_s0 --result 'attempt 1 failed 2026-11-05 22:11 (exit 1, 627 s): --repro temp/e274/sweep_e273_p_s0.json was opened relative to evaluation/ (where _py14.sh runs) after the prediction passes; no card written. Seen in the log only (not a card): kill rule held, 9.1 % of the nodes dropped at thr 0.5, 16.6 % at 0.4; radius curve seed 0 topo 0.618 / 0.624 / 0.618 / 0.238 at 1.0 / 1.25 / 1.75 / 2.0 m' \
  --decision 'no verdict; fixed 2026-11-07 (path against the project root, checked before any GPU work, test added, tests green); to be re-run under the same id in the next GPU window' \
  --notes 'lost the night of 2026-11-05: the launcher exited 1 after the first run; found at 04:23 on 2026-11-06 after a peer message; the computer was shut down on Mirko order at 04:23, so nothing was re-run that night'
