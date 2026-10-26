# HANDOVER - 2026-10-26 03:15

New chat: start in the repository folder (parent of autoresearch/ and this folder), say "Read HANDOVER.md and continue". Rewrite this file at every milestone (<= 3 KB).

## Who does what
- **Lead of loop 19 and owner of the Spurmitte experiments, code, PLAN, schedule, HANDOVER: the data chat "BEV Fusion Spurmitte"** (prompt: experiment_execution/history/prompts/PROGRAM_loop19.md; STOP_AT 2026-10-26 04:00, TEST closed, no mail, no shutdown). The tools chat "BEV Fusion Spurmitte Experiments" owns `autoresearch/tools` (D15 fixes done), evaluation tool code, cleanup. "BEV system" chats work on bev_fusion/ and run many CPU/GPU tests at the same time.

## Where we are (03:15)
- Loop 19 results are in PLAN.md (status column) and in the cards: E236 TIE, E237 TIE, E238 KEEP, E240 TIE (3 pairs: lanes found +0.060, Q +0.006, seed SD Q about 0.003), E241 pair 2 (lanes-found effect full size at 25 %), E244 arm b cut as lever, arm a cut at 3200 / 4800 (partial reading negative: Q 0.673 vs 0.730 at matched vertices), E245 gate passed, E246 killed, E243 invalid, E247 not run, E239 waits for the b456 loader.
- D10/D14 decided: new training uses the CPU-exact distance map (`LCR_DIST_DIR=~/exp_gcar_b123x/feats`); the default of `DIST_DIR` in `training/data/p16_data.py` was switched to the exact map at 03:30 (`LCR_DIST_DIR=~/exp_crn_b123/feats` selects the stored one; test `temp/loop19/test_dist_switch.py` PASS, golden check and spec tests PASS); the foundation network is not retrained.
- No GPU job queued. The loop-19 queue was cut at 02:33 because other chats held the machine (load average 30-64): pair 3 of E240, E244.a1, E244.b1 were never started.
- PLAN.md section "What the system still needs" lists N1-N12 (first: no machine lease between chats, timing and training need a quiet machine).

## Next
1. (done 03:30) DIST_DIR default switch; runs after it train on the exact map: a new arm needs a fresh equal-budget control on the same map (the stored-map controls differ by about 0.002 Q, E237).
2. On a quiet machine: E243 repeat (8 replicates), E240 pair 3, E244 arm a with a tau calibration, E241 pair 3.
3. Add K192-K195 to `literature/register.md`; b456 loader (step 7); `ece` metric for E245.

## Decisions waiting for the user
DECISIONS.md: D4 fast variant (E240/E236 now read: j1rc buys lanes found, not Q), D5 purge, D12 (git rules into METHOD.md). D14 decided 2026-10-26.

## Pointers
PROBLEM.md; METHOD.md; closed_questions.md; literature/OVERVIEW.md; old state history/; git: one repository, tag `lcr/<id>` per run, remote origin = github.com/Archimedes79/Auto-research (pushes only on request).
