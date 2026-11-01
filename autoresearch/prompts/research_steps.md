# Research steps (P1 + P2) - started by hand, one at a time

The outer loop is run by the person: paste one step, read its result, then start the next (or stop). Every step reads `PROBLEM.md`, `HANDOVER.md` and the closed-questions file first, and ends by rewriting `HANDOVER.md`. No downloads (abstract and project pages only) unless the step says so. Method: `autoresearch/METHOD.md`.

**1. Research a group** - `Research group <id> (<budget: n searches, n notes>).`
Spawn `ar-researcher` (mode scan, one per topic, in parallel) with a brief: the group's question from `PROBLEM.md`, the ids already known, what to look for and what to ignore. Then `ar-researcher` (mode deep) for at most <n> of the best notes. Then `ar-researcher` (mode summary) for the group. Show me the digest lines.

**2. Research a foundation topic** - `Research foundations F<n> <topic>.`
As step 1, for a foundation topic of `PROBLEM.md` (building blocks, training and losses, formulations).

**3. Hypotheses** - `Hypotheses with lenses <past critic, one outside lens, system review | all six>.`
Spawn `ar-planner` (mode branches) once per lens, in parallel, on the same inputs (the group digests, the best metric card, the experiment summaries, the ledger). Show me one line per card.

**4. Referee and plan** - `Referee and plan loop <NN> (<GPU budget>).`
`ar-critic` (mode cards) on all cards, then `ar-planner` (mode plan: loop plan + schedule). Show me the tree, the budget, the system improvements, and what needs my decision (TEST, long runs) before the loop starts.

**Light cycle by default**: step 1 only for groups whose gap or literature changed, step 3 with three lenses; all groups and all six lenses every third cycle.
