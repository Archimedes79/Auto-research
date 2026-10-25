# Loop prompt (P3 + P4) - paste after `/loop` and fill the <...>

STOP_AT: <day time> (a work window: keep going until then) | SHUTDOWN: <yes / no; only if no other chat runs> | MAIL: <no / address and how> | TEST: only inside a champion decision recorded in the decisions file

Problem folder: <path>. Loop <NN>. Plan, schedule, ledger, decisions: as named in `HANDOVER.md`. Method: `autoresearch/METHOD.md` P3 and P4. This chat is the lead and owns the code for the loop (say so in `HANDOVER.md`). Do not re-plan; a change of direction goes through `ar-planner` as a new schedule version.

Every wake-up (when a run record changes, else every 20-25 min)
- `wsl python3 ../autoresearch/tools/tick.py --handover` (in the WSL, the namespace of the launchers): it reconciles the run records, starts the ready train / evaluate items within the GPU and CPU slots (their commands come from the schedule's `command` column or the experiment file's arm) and writes the "what runs" block into HANDOVER; read its output, `ledger.py tail --loop <NN>` and the open decisions
- do by hand what the tick lists as "needs the lead": detail plan (`ar-planner`), code (`ar-coder`; `ar-reviewer` mode code for shared code), decide (`tools/decide.py --experiment E<id> --write` when the cards of an experiment are complete), summary (section 3 of the card's `experiment.md`); a run needs its card (experiment file or planned ledger row), run.py refuses otherwise
- at most <n> GPU jobs; checkpoint and resume; never kill another chat's jobs
- write schedule, ledger notes, plan status, HANDOVER (<= 3 KB, rewrite); next wake-up

At STOP_AT (P4)
- end-of-loop section of the plan; report from the experiment summaries; `tools/check_numbers.py`; `ar-reviewer` mode numbers
- card statuses, closed questions, open points for the next cycle; decisions for the person into the decisions file
- commit the experiment code (scratch included), then `tools/clean_temp.py loop<NN> --move`; `tools/check_structure.py`; tests
- HANDOVER rewritten; mail / shutdown only as allowed above
