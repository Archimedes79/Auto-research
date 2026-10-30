# Loop 21 prompt (2026-10-30, written from the chat "create a card and run the loop")

STOP_AT: none; the loop ends when the planned iterations are done and written up | SHUTDOWN: no | MAIL: no | TEST: closed | GPU: none while the BEV-system chat's t2 lane runs (it holds 12 GB and 100 % utilisation at 2026-10-30 01:00); CPU only, 2 threads, no timing claims on a loaded machine

Problem folder: bev_fusion_lane_center/. Loop 21. State: HANDOVER.md, PLAN.md, schedule, ledger, DECISIONS. Method: `autoresearch/METHOD.md` P3 / P4, D16 tools.

Mirko's instruction (chat): a general loop to improve performance (quality and speed), freedom to choose the questions, bullet style; "can you create a card and then run the loop?"

Reading:
- goal: make the lane-graph system better and faster by any route; look at the whole chain; few solid results over many weak ones
- freedom: choose questions from the state files, closed questions and what the numbers show; question the metric and the method too; drop a line as soon as the evidence says so; propose ideas outside the plan with a card
- card: 5 to 10 lines before every experiment that can change a number of the report (question, numeric prediction, rule, kill); `--no-card` only for smoke tests; referee for anything that could change a decision, skipped for cheap CPU follow-ups (stated in the card)
- constraint of the day: the GPU is full (the other chat's lane), so the iterations are CPU-only on cached raw outputs; GPU candidates are listed in PLAN.md and wait
- all numbers VAL 873 via the project's metric code; every comparison states vertices per 100 m; noise band 0.010 for Q, seed SD of Q 0.0025

Fixed rules (unchanged): decide by the written rule through `decide.py`, `run.py` for runs, fresh numbers reviewer before a report, optimisations reproduce the reference graphs, honest negatives, nothing deleted, commit by path, no pushes, no downloads.

End: tests (`run_all_tests.sh`), `check_structure.py`, handover, commit; a short report (HTML builder + PDF) only if Mirko asks.
