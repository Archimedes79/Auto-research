# Loop plan - loop NN (copy to `experiment_execution/PLAN.md` BEFORE the loop starts)

Rule: a loop owns one or two levels of the experiment tree. The plan is the contract of the loop; during the loop only the status column changes.

## Pre-loop checklist (do this in the planning chat, not in the loop)
1. Gap check: what the ledger / `experiments/` say was done; what the literature open list says is missing; the weak points of the last report.
2. One question for the loop (one sentence) and the levels it owns (1 direction, 2 module idea, 3 variant, 4 tuning).
3. For every node: hypothesis that can be false, ONE change, gate and decision rule with a number and the noise band, cost (GPU hours, memory), kill criterion, controls (null, matched reference, strong simple baseline, seed twin).
4. Order by information per GPU hour; successive halving (cheap rung on VAL250 / 1 u, then the full VAL / 4-8 u only for the survivors).
5. Resources: at most 2-3 GPU jobs, < 11 GB; list the data / caches each node needs and whether they exist.
6. Permissions in `PROGRAM.md`: STOP_AT, TEST (how often), mail, shutdown.
7. Create the experiment folders `experiments/E<id>_<slug>/` with `card.md` (from `experiment_card.md`) for every node of the first round; ledger rows `planned`.

## Question of the loop

## Tree
| node | idea | hypothesis | gate | cost | status | experiments |
|---|---|---|---|---|---|---|

## Controls and statistics
- reference / null / seed twin:
- split and TEST use:
- reported with: S, Q, ms per frame

## End of loop
Update the table, ledger, STATE.md, HANDOVER.md; file every experiment; move this file to `history/plans/PLAN_loop<NN>.md`.
