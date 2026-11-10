# studies - scripts that measure something and print numbers

A study answers a question (does X work, how much of Y survives); a gate in `../tests/` checks a known answer and fails with exit code 1. A study that is run again for a number goes through `run.py` like any other run; these ones are the early synthetic ones.

| folder | what | read |
|---|---|---|
| `early/` | the 22 scripts of October 2026 (20 studies, the data of one of them, and `temporal.py`, the module seven of them import; flow, box head, radar encoding, temporal block, camera-only / radar-only decoders, S1 pipeline) that were named `test_*` in `tests/` until D48 (2026-11-03) | `early/README.md` (question, result, who imports it) |
| `recipes/` | the 68 scripts and small result files that made the numbers of the closed experiments t1 - t8 and of the plan (calibration audit, queue drivers, lane probe, SOA gap, radar statistics, FLOPs, profiles), moved out of `temp/` on 2026-11-10 | `recipes/README.md` (one row per folder: experiment, status) |
