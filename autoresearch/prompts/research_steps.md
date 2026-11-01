# Round steps (P1 on a trigger, P2) - started by hand, one at a time

The outer loop is run by the person: paste one step, read its result, then start the next (or stop). Every step reads `PROBLEM.md`, `HANDOVER.md` and the closed-questions file first, and ends by rewriting `HANDOVER.md`. No downloads (abstract and project pages only) unless the step says so. Method: `autoresearch/METHOD.md`.

**0. Start a round** - `Start round <NN>.`
Read `HANDOVER.md`, the gaps and current best of `PROBLEM.md`, the closed-questions file, the open branches at the end of the last `PLAN.md` and `ledger.py tail`. Answer in about ten lines: the question of the round; which research trigger fired (none is the normal case: `METHOD.md`, "Planning a round"); the lenses for step 3; what starts first. Wait for my OK.

**1. Research a group (only when its trigger fired)** - `Research group <id> (<budget: n searches, n notes>).`
Spawn `ar-researcher` (mode scan, one per topic, in parallel) with a brief: the group's question from `PROBLEM.md`, the ids already known, what to look for and what to ignore. Then `ar-researcher` (mode deep) for at most <n> of the best notes. Then `ar-researcher` (mode summary) for the group. Show me the digest lines.

**2. Research a foundation topic (only when its trigger fired)** - `Research foundations F<n> <topic>.`
As step 1, for a foundation topic of `PROBLEM.md` (building blocks, training and losses, formulations).

**3. Hypotheses** - `Hypotheses with lenses <past critic, one outside lens, system review | all six>.`
Spawn `ar-planner` (mode branches) once per lens, in parallel, on the same inputs (the group digests, the best metric card, the experiment summaries, the ledger). Show me one line per card.

**4. Referee and plan** - `Referee and plan loop <NN> (<GPU budget>).`
`ar-critic` (mode cards) on all cards; save its report as a file next to the plan (e.g. `experiment_execution/history/critic_cards_loop<NN>.md`); then `ar-planner` (mode plan) with that file named in the brief: verdicts into the cards' Referee lines, loop plan, schedule, planned ledger rows, the card block per card the tools decide. Show me the tree, the budget, the system improvements, and what needs my decision (TEST, long runs) before the loop starts.

**Default**: steps 1 and 2 only for a topic whose trigger fired (`METHOD.md`, "Planning a round"); step 3 with two or three lenses (the past critic plus one or two others), all six only after two flat rounds.
