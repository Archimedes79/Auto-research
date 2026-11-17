# E2E token champion: one token head on the predicted nodes of the frozen lane head B' (source + tests, no weights)

A winner folder: the source of the system, small enough to hand over; **no weights**. Written by `winner.py` on 2026-11-17 from commit `cacdcd665b`.

## What is here

`code/` holds 8 file(s), 333731 bytes. `MANIFEST.json` lists every file with its sha256; `python winner.py verify <this folder>` re-checks them.

## Numbers to reproduce

CHAMPION (E373): the head trained with the none target and none class at weight 0.25 (--none-weight 0.25), read at link p >= 0.2, node p_none < 0.6, none gate off (chosen on the TRAIN sample). VAL 873 frames, two fresh head seeds (2, 3): topo F1 0.6698 / 0.6769 (cache 0 / 1), F1@1.0 0.8456 / 0.8461, lanes found 0.7274 / 0.7267; floors 0.6359 / 0.6349, plain lane head B 0.6259 / 0.6249; +0.044 / +0.052 over the plain head. RUNNER-UP (E366, E371): none weight 1.0, link 0.1: 0.6617 / 0.6767 and 0.6618 / 0.6733. Whole path 16.6 - 16.8 ms (quiet window, 24 frames, batch 1, bf16) against 19.7 ms for E324 s; 2.323 M parameters (B' 2.196 M + head 0.127 M). Costs: precision 0.68, dup_frac 0.058 / 0.061, 27.4 - 27.8 vertices per 100 m, junction F1 0.20 (below the plain head's 0.27). No TEST number. Status: stage S1, B' frozen; VAL has been read by every arm of the night.

## How to run

```
NEEDS the repository (it imports jtok, jtok_eval, qgraph_eval, qgraph_readout, qtargets, trainer): copy code/ over bev_fusion_lane_center/ and run from there. Train the champion: bash tools/env/_py14.sh jtoken_train.py --name e373_cb_s<S> --seed S --ident e292_a --none-weight 0.25 --steps 12000 --bs 8 --lr 1e-3 --workers 4 (the runner-up: the same without --none-weight). Read out: bash tools/env/_py14.sh jtoken_eval.py dump --run e373_cb_s<S> --seed S; jtoken_eval.py score --run e373_cb_s<S> --seed S --link-thr 0.2 --keep-thr 0.6 --no-none-gate. Time: jtoken_eval.py timing --run e365_s1_s0 --b-run runs/e292_a_s0. Tests: bash tools/env/_py14.sh training/grid/test_jtoken.py, training/grid/test_lane_tokcache.py, evaluation/tests/test_jtoken_eval.py (all exit 0 on 2026-11-17).
```

## Weights

champion heads: runs/e373_cb_s0 (cache 0, head seed 2), runs/e373_cb_s1 (cache 1, head seed 3); selection heads runs/e367_b25_s0 / s1; runner-up runs/e365_s1_s0 / s1, runs/e371_cf_s0 / s1; frozen lane head B': runs/e292_a_s0 / s1; token cache runs/e365_cache/be292_a_{0,1}_K512_9ad1d4ce30_v1 (derived). All in the repository runs, not in git.
