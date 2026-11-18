# E2E token champion, two stages (D120): one token head on the predicted nodes of the frozen lane head B', then trained together with B's node head (S4, E372); source + tests, no weights

A winner folder: the source of the system, small enough to hand over; **no weights**. Written by `winner.py` on 2026-11-18 from commit `78713c4bd7`.

## What is here

`code/` holds 13 file(s), 689716 bytes. `MANIFEST.json` lists every file with its sha256; `python winner.py verify <this folder>` re-checks them.

## Numbers to reproduce

CHAMPION since D120 (E372 S4-HOLDS): VAL 873 frames, topo F1 0.6682 / 0.6813 (s0 / s1) at the D115 cell (short links 0.3 / long links 0.2 / keep 0.6 / gate off); against the same head with B' frozen -0.0003 / +0.0049, against the staged champion (stage 1 alone, 0.6679 / 0.6762) +0.0002 / +0.0051; junction F1 s0 0.2538. Inference path 16.21 ms quiet (batch 1, bf16, RTX 3060), 2.323 M parameters (B' 2.196 M + head 0.127 M), unchanged by stage 2. The landed-tree re-read of seed 0 is bit for bit. No TEST number.

## How to run

```
NEEDS the repository (it imports qgraph, qpredict, qtargets, qgraph_eval, trainer, jtok): copy code/ over bev_fusion_lane_center/ and run from there. Stage 1: bash tools/env/_py14.sh jtoken_train.py --name e373_cb_s<S> --seed S --ident e292_a --none-weight 0.25 --steps 12000 --bs 8 --lr 1e-3 --workers 4. Stage 2: bash tools/env/_py14.sh jlive_train.py --head-run e373_cb_s<S> --train-nodes 1 --live-inputs 1 --steps 3000 --bs 8 --lr 1e-4 (name the run e372_s4_s<S>). Read out LIVE: jtoken_eval.py dump --live --run e372_s4_s<S> --seed S; jtoken_eval.py score --live --run e372_s4_s<S> --seed S --link-thr 0.3 --link-thr-long 0.2 --keep-thr 0.6 --no-none-gate. Tests: bash tools/env/_py14.sh training/grid/test_jtoken.py, test_lane_tokcache.py, test_lane_live.py, test_jlive_train.py, evaluation/tests/test_jtoken_eval.py.
```

## Weights

runs/e372_s4_s0 / s1 (head + node head of B', b_state); stage 1: runs/e373_cb_s0 / s1; frozen lane head B': runs/e292_a_s0 / s1; token cache runs/e365_cache (derived). All in the repository runs, not in git.
