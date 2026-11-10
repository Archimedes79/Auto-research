# Round steps (P1 on a trigger, P2) - started by hand, one at a time

The outer loop is run by the person: paste one step, read its result, then start the next (or stop). Every step reads `PROBLEM.md`, `HANDOVER.md` and the closed-questions file first, and ends by rewriting `HANDOVER.md`. No downloads (abstract and project pages only) unless the step says so. Method: `autoresearch/METHOD.md`.

**0. Start a round** - `Start round <NN>.`
Read `HANDOVER.md`, the gaps and current best of `PROBLEM.md`, the closed-questions file, the open branches at the end of the last `PLAN.md` and `ledger.py tail`. Answer in about ten lines: the question of the round; which research trigger fired (none is the normal case: `METHOD.md`, "Planning a round"); the lenses for step 3; what starts first. Wait for my OK.

**1. Research a topic or a group (when its trigger fired, or when I order it)** - `Research <topic, or group id, or foundation F<n>> (<budget: n searches, n notes>).`
One order, three runs of `ar-researcher` in a row without waiting for me: mode scan (one per subtopic, in parallel) with a brief that gives the question (from `PROBLEM.md`, or my words), the ids already known, what to look for and what to ignore; then mode deep for at most <n> of the best notes; then mode summary, which writes the overview of `templates/group_summary.md` as `<research folder>/topics/<slug>.md` (a group: its summary file): the digest, the techniques with K ids, a table "For experiment planning" (candidate, gap it attacks, variant or block, cost, first test) and a link to every note and extraction it rests on. Show me the digest lines and the link to the overview. That page is the input of step 3 and of any card that names a technique; nothing else has to be read.

**3. Hypotheses** - `Hypotheses with lenses <past critic, one outside lens, system review | all six>.`
Spawn `ar-planner` (mode branches) once per lens, in parallel, on the same inputs (the group digests, the best metric card, the experiment summaries, the ledger). Show me one line per card.

**4. Referee and plan** - `Referee and plan loop <NN> (<GPU budget>).`
`ar-critic` (mode cards) on all cards; save its report as a file next to the plan (e.g. `experiment_execution/history/critic_cards_loop<NN>.md`); then `ar-planner` (mode plan) with that file named in the brief: verdicts into the cards' Referee lines, loop plan, schedule, planned ledger rows, the card block per card the tools decide. Show me the tree, the budget, the system improvements, and what needs my decision (TEST, long runs) before the loop starts.

**Default**: step 1 only for a topic whose trigger fired or that I order (step 2 was merged into step 1 on 2026-11-10; the numbers stay) (`METHOD.md`, "Planning a round"); step 3 with two or three lenses (the past critic plus one or two others), all six only after two flat rounds.
