# Plan - loop <NN>

Written in P2 by ar-planner, stable during the loop. It holds the reasoning and the tree, no status: `wsl python3 autoresearch/tools/plan.py status` reads the status of each node from the schedule and the ledger. The schedule (what runs next) is the problem's schedule file, kept by the schedule tool.

Question of the loop: <one sentence>. Levels this loop owns: <e.g. 2 = module idea, 3 = variant>. Budget: <GPU hours, comparisons>.

## Tree
| node (= the experiment id, E<n>: schedule items name it in their `node` column) | experiment | hypothesis (short) | decision rule | cost | priority |
|---|---|---|---|---|---|

## System improvements
| item | lack in the software | change | done when | status |
|---|---|---|---|---|

## Controls and statistics
matched reference, null baseline, equal-budget control, seed twin; split rules; TEST only inside a champion decision.

## End of loop (P4)
- per node: result and decision (one line each, numbers from metric cards)
- best system now, and what changed against the last best
- questions closed (into the closed-questions file), open points for the next cycle
- decisions for the person (into the decisions folder, one file `decisions/D<n>.md` each)
- checks: numbers check, numbers reviewer, structure check, tests
