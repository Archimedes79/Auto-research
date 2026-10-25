# Loop 19 prompt (from autoresearch/prompts/loop.md, filled 2026-10-25 21:40)

STOP_AT: 2026-10-26 04:00 (a work window: keep going until then) | SHUTDOWN: no | MAIL: no | TEST: only inside a champion decision recorded in the decisions file (none exists: TEST stays closed)

Problem folder: bev_fusion_lane_center/. Loop 19. Plan, schedule, ledger, decisions: as named in `HANDOVER.md`. Method: `autoresearch/METHOD.md` P3 and P4. This chat (the data chat "BEV Fusion Spurmitte") is the lead and owns the code of the loop and HANDOVER / PLAN / schedule; the tools chat ("BEV Fusion Spurmitte Experiments") owns `autoresearch/tools`, the evaluation tool code and cleanup, and starts no GPU job without telling the lead. Do not re-plan; a change of direction goes through `ar-planner` as a new schedule version.

Mirko's instruction (chat, 2026-10-25 about 21:09, /loop): "Run the full test suite on lane center detection till 4 o clock in the morning. Use past loop descriptions but the new system. Use existing BEV networks here. Generate experiment and fix data. What needs to be improved in the system. What results do we get?"

Reading of it:
- full test suite = `tools/maintenance/run_all_tests.sh` at the start and at the end (layer 0-1), then the cycle-19 experiments (layers 3-4: VAL 873, 2 seeds, paired scene bootstrap)
- existing BEV networks = the registry models (champion, j1 / j1rc / j1eq family, control E94) and their frozen GaussianCaR features; no new backbone
- generate experiment = the cycle-19 cards E236-E247 (referee G2 of tonight); fix data = D10 (exact distance map, E251) and, if time allows, the b456 loader (step 7)
- results and system improvements = experiment summaries, the loop table of `metrics.py`, and an end-of-loop list of what the system needs

Every wake-up (when a run record changes, else every 20-25 min)
- `wsl python3 ../autoresearch/tools/tick.py --handover`, then `ledger.py tail --loop 19`; one GPU training at a time (the 786 s solo vs 1,500 s paired measurement of the referee), CPU jobs beside it; GPU < 11 GB
- do by hand what the tick lists as "needs the lead"; shared-code edits in two reviewer passes (mode code)
- write schedule, ledger notes, plan status, HANDOVER (<= 3 KB, rewrite); next wake-up

At STOP_AT (P4)
- end-of-loop section of the plan; report from the experiment summaries; `check_numbers.py`; `ar-reviewer` mode numbers
- the system-improvement list (tools, data, method) as a section of the plan and as DECISIONS entries where a decision is Mirko's
- commit the code and records, `clean_temp.py loop19 --move` only if the scratch is filed; `check_structure.py`; `run_all_tests.sh`
- HANDOVER rewritten
