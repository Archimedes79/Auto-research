# HANDOVER - 2026-10-25 22:00

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 19 and owner of the Spurmitte experiments, code, PLAN, schedule, HANDOVER: the data chat "BEV Fusion Spurmitte"** (prompt: experiment_execution/history/prompts/PROGRAM_loop19.md; STOP_AT 2026-10-26 04:00, TEST closed, no mail, no shutdown). The tools chat "BEV Fusion Spurmitte Experiments" owns `autoresearch/tools`, the evaluation tool code and cleanup, and starts no GPU job without telling the lead. "BEV system" works on bev_fusion/ and literature.

## Where we are (22:00)
- Loop 18 closed. Loop 19 = cards E236-E247 (refereed tonight, verdicts in the cards), PLAN.md and schedule.csv by ar-planner. Start: full test suite 14/14 PASS; D10 data fix done (E251.dist/.chk: exact distance map in ~/exp_gcar_b123x/feats; stored map damaged in 51 % of the train frames); shared code (default off, reviewed): p16_train `--save_at`, p16_data `LCR_DIST_DIR`, p16_loss `P19_REC_CLAMP`.
- Done: E243 timing (session invalid: machine +3 ms, spread > 1.5 ms; relative order j1 - f1 = -0.5 ms; repeat on a quiet machine), E237 step 0 (8.7 % of the supervised pixels off by > 0.25 m).
- Running, GPU one training at a time (about 30 min each): E240.a2 -> c2 -> E237.a0 -> a1 -> E244.b0 -> E240.a3 -> c3 -> ev0 (launchers wait with --after); CPU: E236 tau sweep.

## Next
1. Wake-up: `wsl python3 ../autoresearch/tools/tick.py --handover`, `ledger.py tail --loop 19`; decide with `decide.py --experiment`; summaries into section 3 of each card.
2. Open work: E236 matched table (`run_e236.sh matched`), E244 dead-zone diagnostic + phd loss (block), E238 `--map_seed`, E245/E246/E247 step 0, E243 repeat; E239 waits for the b456 loader (step 7).
3. 03:00-04:00: tests, check_numbers, ar-reviewer numbers, check_structure, plan section "What the system still needs", this file.

## Decisions waiting for the user
DECISIONS.md: D4 fast variant, D5 purge, D9 (done in P5), D12 (git rules into METHOD.md); D10 decided (a) by Mirko 20:45.

## Pointers
PROBLEM.md; METHOD.md; closed_questions.md; literature/OVERVIEW.md; old state history/; git: one repository, tag `lcr/<id>` per run (E240.a2 tagged by hand: snapshot race), remote origin = github.com/Archimedes79/Auto-research (pushes only on request).
