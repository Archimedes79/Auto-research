# E324 s: the end-to-end lane graph (inference package, no weights)

A winner folder: the source of the system, small enough to hand over; **no weights**. Written by `winner.py` on 2026-11-17 from commit `ac3c387378`.

## What is here

`code/` holds 19 file(s), 89782 bytes. `MANIFEST.json` lists every file with its sha256; `python winner.py verify <this folder>` re-checks them.

## Numbers to reproduce

VAL 873 frames, topo F1 seed 0 / seed 1: end-to-end read-out with --attached-ends 0.6912 / 0.6948 (rule wiring); the default read-out (snapped, with a rule step) 0.6981 / 0.7010. The learned wiring head (E358, 0.6931 / 0.6964) is not in this package: see `winners/e358_learned_wiring/`.

## How to run

```
bash tools/env/_py14.sh training/lanegraph/scripts/lg_infer.py --seed 0 --first 20 --attached-ends --out /abs/path/graphs.json   # run from bev_fusion_lane_center/; tests: training/lanegraph/tests/run_lg_tests.py
```

## Weights

runs/e324_s_s0 and runs/e324_s_s1 (the junction system) and runs/e273_p_s0 / s1 (lane head B): in the repository runs, not in git
