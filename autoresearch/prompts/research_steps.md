# Round steps (P1 on a trigger, P2) - started by hand, one at a time

The outer loop is run by the person: paste one step, read its result, then start the next (or stop). Every step reads `PROBLEM.md`, `HANDOVER.md` and the closed-questions file first, and ends by rewriting `HANDOVER.md`. No downloads (abstract, project and repository pages only) unless the step says so (step 1 (b) does). Method: `autoresearch/METHOD.md`.

**0. Start a round** - `Start round <NN>.`
Read `HANDOVER.md`, the gaps and current best of `PROBLEM.md`, the closed-questions file, the open branches at the end of the last `PLAN.md` and `ledger.py tail`. Answer in about ten lines: the question of the round; which research trigger fired (none is the normal case: `METHOD.md`, "Planning a round"); the lenses for step 3; what starts first. Wait for my OK.

**1. Research a topic or a group (when its trigger fired, or when I order it)** - `Research <topic, or group id, or foundation F<n>> (<budget: n searches, n notes>).`
One order, run in a row without waiting for me. The topic folder is `<research folder>/topics/<slug>/`.
(a) `ar-researcher` mode scan (several in parallel, one per subtopic, when the topic is wide) with a brief that gives the question (from `PROBLEM.md`, or my words), the ids already known, what to look for and what to ignore: papers, the foundations they build on, and code. Each scan writes its own `candidates_<subtopic>.md`; the lead merges them into `candidates.md` (one row per paper, a repository that belongs to a paper in that paper's row, a short unique id per row: it is the folder name) and drops the rest of the doubles.
(b) `ar-researcher` mode deep, ONE AGENT PER CANDIDATE, in parallel (at most <n>, the best by the one-line reason): each works in its own subfolder `<id>/` and writes `note.md` (seven header lines: title, relevance 0-3, about three sentences on what it does and why it matters, status, paper link, code link, verdict) plus the deeper material (method, results, code review, port card). This step allows each deep agent to read the full text of its one paper through the page fetch tool and the repository pages (nothing saved, nothing cloned).
(c) The lead runs `python autoresearch/tools/topic_table.py <topic folder>`: it makes `table.md` and `table.csv` from the headers (title, summary, relevance, status, paper, code, best first). A note it names is sent back to its agent, and the tool is run again; it removes any older table when it finds a problem, so a stale table is never read. The table is made by the script, never typed.
(c2) Spot check: `ar-critic` mode numbers, a fresh context, re-opens the sources of the three notes with the highest relevance and one more chosen at random, and compares each summary and each number with its page. A contradicted or unreadable claim sends that note back to its agent (two rounds at most); after that the candidate is struck from `candidates.md` with the reason and its note is moved to `rejected/`.
(d) `ar-researcher` mode summary (it cannot run the script; it reads `table.md`) writes `overview.md` (`templates/group_summary.md`): the digest, the techniques with K ids, the table "For experiment planning" and the links to the generated table and the notes.
Show me the digest lines and the links to `overview.md` and `table.md`. These two pages are the input of step 3 (the planner turns the table into hypotheses) and of any card that names a technique; nothing else has to be read.

Procedure as a contract (each stage starts only when the one before has met its gate):
| stage | who | input | output | gate to the next stage | on failure |
|---|---|---|---|---|---|
| order | the person | topic or group, budget | the lead's brief: question, known ids, budget | the question is one sentence; the known ids are read from the papers table and the K register | the lead asks one question |
| scan | researcher(s), parallel | the brief | `candidates_<subtopic>.md` | at least one row; every row has id, title, kind, paper link, one line why; not more rows than the budget | re-brief once, then stop and report |
| merge | lead | the lists | `candidates.md` | ids unique and folder-safe (letters, digits, . _ -); one row per paper; known papers marked | fix by hand |
| deep | one researcher per candidate, parallel | one row | `<id>/note.md` + extras | header valid (the table tool); the body has the template sections; `status` is true | back to the agent, two rounds, then struck |
| table | the lead (the script) | all notes | `table.md`, `table.csv` | exit 0 | the tool removes old tables and names the problems |
| check | critic (numbers) | 3 top notes + 1 random, with their sources | a short report | no claim contradicted | back to the agent, two rounds, then struck |
| summary | researcher | table, notes, register | `overview.md` | digest of three lines; every technique has a K id with source; every link resolves | back to the agent |
| plan | planner (branches) | overview, table | cards (step 3) | the cards name their sources (K ids, note ids) | step 3 |
Done when: `table.md` exists and was made by the script after the last change of a note, the spot check passed, `overview.md` links resolve, and the lead has shown the digest and the two links.

**3. Hypotheses** - `Hypotheses with lenses <past critic, one outside lens, system review | all six>.`
Spawn `ar-planner` (mode branches) once per lens, in parallel, on the same inputs (the overview and the table of the topic, the best metric card, the experiment summaries, the ledger). Show me one line per card.

**4. Referee and plan** - `Referee and plan loop <NN> (<GPU budget>).`
`ar-critic` (mode cards) on all cards; save its report as a file next to the plan (e.g. `experiment_execution/history/critic_cards_loop<NN>.md`); then `ar-planner` (mode plan) with that file named in the brief: verdicts into the cards' Referee lines, loop plan, schedule, planned ledger rows, the card block per card the tools decide. Show me the tree, the budget, the system improvements, and what needs my decision (TEST, long runs) before the loop starts.

**Default**: step 1 only for a topic whose trigger fired or that I order (step 2 was merged into step 1 on 2026-11-10; the numbers stay) (`METHOD.md`, "Planning a round"); step 3 with two or three lenses (the past critic plus one or two others), all six only after two flat rounds.
