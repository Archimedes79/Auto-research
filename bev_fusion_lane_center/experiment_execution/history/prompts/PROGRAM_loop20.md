# Loop 20 prompt (from autoresearch/prompts/loop.md, filled 2026-10-26 23:55)

STOP_AT: none given; the loop ends when the four iterations, the framework test and the weekly report are done | SHUTDOWN: yes, as the last step, only if no other chat runs a job (CLAUDE.md) | MAIL: no | TEST: only inside a champion decision recorded in the decisions file (none exists: TEST stays closed)

Problem folder: bev_fusion_lane_center/. Loop 20. Plan, schedule, ledger, decisions: as named in `HANDOVER.md`. Method: `autoresearch/METHOD.md` P3 and P4, with the simplified tools of D16 (run state derived from the run records). This chat is the lead and owns the code of the loop.

Mirko's instruction (chat, 2026-10-26 about 23:45): "When the other training is ready, do 4 iterations here. Perform auto research, test the methodology and the framework, improve the framework where necessary. Generate results. Report all results so far for this week in the usual form we have discussed multiple times. Then shutdown the computer."

Reading of it:
- the other training = t1_r3_combined of the bev_fusion chats; finished 2026-10-26 14:58, only an audit probe was running at 23:47 -> ready
- 4 iterations = four cards through the whole cycle (card, referee, run, decide by the written rule, summary), chosen from the open points of loop 19 and from Mirko's GT remarks (map offsets): E252 GT map-offset exposure by city (CPU), E253 timing repeat on a quiet machine, E254 seed pair 3 of the recall fine-tune (completes the seed floor and E241), E255 Poisson-loss arm a, full schedule with a pre-registered matched-vertex rule
- test the methodology and the framework = use the D16 tools as they are (schedule show / next, run.py chains with --after, ledger written at the end, decide.py, experiment.py check, check_numbers, a fresh numbers reviewer) and write down where they hurt; fix what hurts in the project's own scripts; changes of `autoresearch/tools` (protected) only as a DECISIONS entry
- weekly report in the usual form = week 3 (25 - 31 Oct, loops 18 - 20): builder in `reports/build/reports/` (like `build_week2_report.py`), PDF through `print_weekly.ps1`, architecture drawing + numbers table + result images per experiment, setting and noise band named, list of corrected claims, experiment index, one fresh numbers reviewer before it is final
- shutdown = `shutdown /s /t 60` from Windows after the last commit, WSL idle, no other chat running (ListAgents / list_sessions, ps in WSL)
