# Loop prompt (P3 + P4) - paste after `/loop` and fill the <...>

STOP_AT: <day time> (a work window: keep going until then) | SHUTDOWN: <yes / no; only if no other chat runs> | MAIL: <no / address and how> | TEST: <no / once for winner + control>

Problem folder: <path>. Loop <NN>. Plan: `PLAN.md`; schedule: `schedule.csv`; ledger: `LEDGER.csv`. Method: `autoresearch/METHOD.md` P3 and P4. Do not re-plan; a change of direction goes through `ar-scheduler` as a new schedule version.

Every wake-up (20-25 min)
- read HANDOVER, `schedule.py show`, the ledger; reconcile running items with the run records (done / failed / stalled)
- advance ready items in schedule order: detail plan (`ar-detail-planner`) -> code (`ar-coder`, `ar-code-reviewer` for shared code) -> train (`autoresearch/tools/run.py`, commits and tags the code) -> evaluate (the problem's evaluation entry point) -> decide (`tools/decide.py --write`) -> experiment summary
- at most <n> GPU jobs; checkpoint and resume; never kill another chat's jobs
- write schedule, ledger notes, HANDOVER (<= 3 KB, rewrite); next wake-up

At STOP_AT (P4)
- loop summary / report from the experiment summaries; `tools/check_numbers.py`; one `ar-numbers-reviewer`
- card statuses, `closed_questions.md`, open points for the next cycle
- commit the experiment code (scratch included), then `tools/clean_temp.py loop<NN> --move`; `tools/check_structure.py`; tests
- HANDOVER rewritten; mail / shutdown only as allowed above
