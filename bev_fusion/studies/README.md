# studies - scripts that measure something and print numbers

A study answers a question (does X work, how much of Y survives); a gate in `../tests/` checks a known answer and fails with exit code 1. A study that is run again for a number goes through `run.py` like any other run; these ones are the early synthetic ones.

| folder | what | read |
|---|---|---|
| `early/` | the 20 scripts (19 studies and the data of one of them) of October 2026 (flow, box head, radar encoding, temporal block, camera-only / radar-only decoders, S1 pipeline) that were named `test_*` in `tests/` until D48 (2026-11-03) | `early/README.md` (question, result, who imports it) |
