# Loop prompt (P3 + P4) - paste after `/loop` and fill the <...>

STOP_AT: <day time> (a work window: keep going until then) | SHUTDOWN: <yes / no; only if no other chat runs> | MAIL: <no / address and how> | TEST: only inside a champion decision recorded in the decisions file

Problem folder: <path>. Loop <NN>. Plan, schedule, ledger, decisions: as named in `HANDOVER.md`. Method: `autoresearch/METHOD.md` P3 and P4. This chat is the lead and owns the code for the loop (say so in `HANDOVER.md`). Do not re-plan; a change of direction goes through `ar-planner` as a new schedule version.

Every wake-up (20-25 min)
- read HANDOVER, the schedule (`schedule.py show`), the ledger; reconcile running items with the run records (done / failed / stalled)
- advance ready items in schedule order: detail plan (`ar-planner`) -> code (`ar-coder`; `ar-reviewer` mode code for shared code) -> train (`autoresearch/tools/run.py`, commits and tags the code) -> evaluate (the problem's evaluation entry point) -> decide (`tools/decide.py --write`) -> summary (section 3 of the card's `experiment.md`)
- at most <n> GPU jobs; checkpoint and resume; never kill another chat's jobs
- write schedule, ledger notes, plan status, HANDOVER (<= 3 KB, rewrite); next wake-up

At STOP_AT (P4)
- end-of-loop section of the plan; report from the experiment summaries; `tools/check_numbers.py`; `ar-reviewer` mode numbers
- card statuses, closed questions, open points for the next cycle; decisions for the person into the decisions file
- commit the experiment code (scratch included), then `tools/clean_temp.py loop<NN> --move`; `tools/check_structure.py`; tests
- HANDOVER rewritten; mail / shutdown only as allowed above
