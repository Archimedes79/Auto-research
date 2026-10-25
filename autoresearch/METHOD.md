# The autoresearch method

A general workflow for improving a technical system by research, with agents doing most of the work and a person setting direction and granting permissions. It knows nothing about lane graphs; a problem plugs in through one folder with `PROBLEM.md` + `project.json` (see `README.md`).

## Two systems

| | the product | the improver (this method) |
|---|---|---|
| what | the system being improved (here: a lane-centre-line graph from camera + radar with SD-map assignment) | research -> hypotheses -> plans -> experiments -> decisions -> summaries |
| code | the problem's code folders (training, inference, data) | `autoresearch/tools/` (launcher, ledger, decisions, checks, cleanup) + the problem's evaluation code |
| changed by | reviewed commits; every experiment runs a tagged commit | improvements to this folder, proposed by the system-review lens (phase 2) |

## The cycle

```
 P0 FRAME            P1 RESEARCH (hierarchical)          P2 HYPOTHESES + PLAN                 P3 LOOP (per experiment set)         P4 CLOSE
 PROBLEM.md          direction -> topic briefs           lens agents -> hypothesis cards      detail plan -> code -> tag           summaries -> report
 metrics, protocol   scouts -> paper notes (L1)          referee -> accepted cards            -> train (run.py) -> evaluate        feedback: card status,
 current best, gaps  deep readers -> extractions (L2)    planner -> experiment plan           -> decide (rule) -> experiment       closed questions, backlog
 problem groups      group summarizers -> summaries (L3) scheduler -> schedule                summary (open points, next steps)    cleanup temp, handover
                     lead -> research digest (L4)        system improvements                  wake-ups resume from the schedule
```

A cycle is P1-P4. P1 and P2 run before a loop (they can run in the background while a previous loop executes). P3 is the `/loop`. P4 closes it and feeds the next P1 / P2. Each phase ends with a gate (below); a phase reads only the summaries of the phase before it and opens details only for the ids it uses.

### P0 Frame (once per problem, revisited each cycle)
`PROBLEM.md`: goal and constraints, the product chain, the metrics and the evaluation protocol, the current best with its numbers, the known gaps, the **problem groups** (the taxonomy every later artifact is filed under) and the **foundation topics** (building blocks, training and losses, mathematical formulations). The person owns this file.

### P1 Research: from paper details to summaries
| step | who | reads | writes |
|---|---|---|---|
| 1a direction | lead | PROBLEM.md, last digest, open gaps | one **topic brief** per scout: question, what we know (ids), what to look for, what to ignore, budget |
| 1b scouting | one `ar-scout` per topic, in parallel; three families: **state of the art on the problem**, **foundations and new building blocks** (what they are for), **training and losses** | brief, search results, abstract pages | **paper notes (L1)**: what, how, results, relevance per problem group (0-3), transferable technique, cost, code / licence, fact status (checked / digest / memory) |
| 1c deep reading | `ar-deep-reader` for the top notes | paper full text, repository | **extraction (L2)** with a **port card**: interface, what to take, licence, minimal test |
| 1d summarizing | one `ar-group-summarizer` per problem group and per foundation topic | the L1 / L2 of its group | **group summary (L3)**: state of the art and best numbers, techniques ranked by expected transfer (each with a K id in the register), open questions, what not to do |
| 1e digest | lead | all L3 | **research digest (L4)**, one page: per group the top 3 impulses and why |

Gate G1: every group has a summary; every technique named in a summary has a K id with a source and a fact status.

### P2 Hypotheses and plan
Lens agents work in parallel on the same inputs: the research digest (L4), the experiment summaries and ledger of past cycles, the current metric card of the best system, the system review. Each writes **hypothesis cards** (template `templates/hypothesis_card.md`).

| lens | question it asks |
|---|---|
| **state of the art** | what do the best methods on this problem do that we do not, and which of it fits our constraints? |
| **foundations** | which building block, structure or mathematical formulation would express the problem better (e.g. a different output parametrisation, memory, attention pattern, optimisation view)? |
| **training and losses** | which training regime or loss would remove a known failure (curriculum, matching-free losses, distillation, calibration, schedule)? |
| **past critic** (`ar-past-critic`) | reviews past experiments critically: conclusions drawn too early, confounds, missing controls, untested assumptions, contradictions between results, repeated negatives; re-opens or closes questions |
| **system review** (`ar-system-reviewer`) | where does the software foundation lack (speed, checks, reuse, missing tools, provenance)? writes system-improvement cards |
| **free** | any idea the agent believes in, argued like the others |

Then:
- `ar-referee` challenges every card: falsifiable? mechanism plausible? already tried (ledger, closed questions)? confounded? cost realistic? -> accept / revise / reject with a reason.
- `ar-planner` deduplicates, clusters and ranks the accepted cards by value (expected gain x probability / cost), and builds the **experiment plan** (`templates/experiment_plan.md`). The plan is a tree of the levels this cycle owns, with nodes ready for detail planning, and a separate list of system improvements.
- `ar-scheduler` turns the plan into the **schedule** (`templates/schedule.csv`): one row per work item with state, dependencies, resource class (GPU / CPU / agent), budget, priority, owner, and a checkpoint policy. The schedule is the only place that says what happens next, so a disrupted loop resumes from it.

Gate G2 (hand-over ready, checked by the referee): a card is ready when it has a falsifiable statement, a mechanism, a prediction with a number and a metric, a decision rule with the noise band, controls (null, matched reference, equal budget, seed twin), a cost estimate, a kill criterion, its source ids, and its inputs either ready or listed as tasks.

### P3 Loop: detail planning and execution (one experiment set)
For every scheduled card, in order of the schedule:

1. **Detail plan** (`ar-detail-planner`, `templates/detail_plan.md`): understand the card, then actionable steps:
   - coding: which block or change, where, which spec test proves it works;
   - training: configs, seeds, budget, checkpoint and resume;
   - evaluation: which registry names, splits, comparisons, metrics, the rule;
   - result summary: what the table and images must show.
   It writes the experiment folder with the card, adds ledger rows (`planned`) and schedule items.
2. **Code** (`ar-coder`): implements the change with its test. A change of the core bundle (code other experiments use) gets an `ar-code-reviewer` pass on the diff before it is used. The launcher commits and tags the code when the run starts (`<prefix>/E<id>.<arm>`).
3. **Train**: `tools/run.py` (ledger row, commit + tag, prediction and rule, retries, done flag only on success), resumable training.
4. **Evaluate**: the problem's evaluation entry point (one command per named model; a card with all metrics and the commit).
5. **Decide**: `tools/decide.py` applies the written rule (KEEP / CUT / TIE / INCONCLUSIVE); INCONCLUSIVE adds a seed instead of adopting.
6. **Summarise** (`templates/experiment_summary.md`): numbers with controls, images, decision, weaknesses, open points, next steps -> the experiment folder; the card's status in the plan.

Wake-up routine of the loop: read HANDOVER + schedule + ledger -> reconcile (what finished, failed, stalled) -> advance the next ready items -> write schedule, ledger, HANDOVER -> next wake-up. Nothing else is state.

Gate G3 (per experiment): every number comes from a card or a run record; the rule was applied as written; the summary has open points and next steps.

### P4 Close
- **Loop summary / report** from the experiment summaries and cards; `tools/check_numbers.py` on the text; one fresh `ar-numbers-reviewer` against primary sources.
- **Feedback**: each card gets a status (confirmed / refuted / open / closed); `closed_questions.md` gets every question answered twice the same way; the backlog of the next P2 gets the open points.
- **Cleanup**: commit the experiment code (scratch included) so the tags keep it, then `tools/clean_temp.py loopNN --move`; old state into history; `tools/check_structure.py`; rewrite HANDOVER (<= 3 KB).

Gate G4: tests pass, structure check clean, every run of the loop has a ledger decision and a summary, HANDOVER rewritten.

## Disruptions
A loop can be stopped at any moment (user, crash, shutdown, another chat). The rules:
- **State in a few files**: HANDOVER (narrative, <= 3 KB, names the lead chat that owns the code), the plan, the schedule (what next), the ledger (what ran), the decisions file (what waits for the person). A new chat continues from these alone.
- **Reading**: single-file records (ledger, decisions, closed questions, plan) are complete files that agents search; experiment folders are read in full for the last two loops, older loops through their loop / weekly summaries and the problem's current best.
- **Idempotent steps**: every schedule item can be restarted; trainings checkpoint and resume; a run id that exists is resumed, not duplicated; a done flag means exit code 0.
- **Reconcile first**: after a disruption the lead marks running items as done / failed / stalled from the run records, then continues. A user change of direction goes through the planner (a new schedule version), not into ad-hoc runs.
- **STOP_AT is a work window**: when the schedule is empty early, the scheduler pulls the next items (controls, seeds, robustness, backlog) with rules written before the runs.

## Decisions of the person
The person decides through one decisions file (`templates/decisions.md`), never through file edits by agents: promoting a champion (which includes the one TEST check of the candidate and the current champion; TEST is never used otherwise), changes of `PROBLEM.md`, changes of the tools or of the gates of this method, and permissions (downloads, installs, mail, shutdown, backups, remote). An agent writes the entry with options, recommendation and evidence, and acts only after the answer is recorded.

## Code and git
- The repository holds code and text records: this folder, the problem's code folders and scratch scripts (so a tag keeps them), the rules and templates, and the text records (plan, schedule, ledger, decisions, run records, cards, experiment summaries, metric cards, literature summaries). Never data, checkpoints, caches or binaries.
- Git history is the long-term memory; nothing committed is rolled away. Only what an agent reads rolls (see Disruptions, "Reading").
- `main` holds the reviewed core bundle; one lead chat per loop owns the code; other chats work in research or reports or only read (separate worktrees need the state outside the checkout first).
- Every run commits the current code and tags it `<prefix>/E<id>.<arm>`; the ledger and the metric card record the commit. A result can always be traced to the exact code, and scratch scripts can be deleted after the loop because the tag keeps them.

## Roles at a glance
| agent | phase | input -> output |
|---|---|---|
| lead (main chat) | all | direction, briefs, digest, wake-ups, decisions with the person |
| `ar-scout` | P1 | topic brief -> paper notes (L1) |
| `ar-deep-reader` | P1 | top notes -> extraction + port card (L2) |
| `ar-group-summarizer` | P1 | notes of a group -> group summary (L3) |
| `ar-hypothesis` (lens given in the brief) | P2 | digest + past + card -> hypothesis cards |
| `ar-past-critic` | P2 | ledger + experiment summaries -> critique + hypothesis cards |
| `ar-system-reviewer` | P2 | code base + run records -> system-improvement cards |
| `ar-referee` | P2 | cards -> verdicts, hand-over readiness |
| `ar-planner` | P2 | accepted cards -> experiment plan |
| `ar-scheduler` | P2 / P3 | plan -> schedule; re-plans after a disruption |
| `ar-detail-planner` | P3 | card -> detail plan, experiment folder, ledger rows |
| `ar-coder` | P3 | detail plan -> code + test |
| `ar-code-reviewer` | P3 | diff of the core bundle -> findings |
| `ar-numbers-reviewer` | P4 | report + primary sources -> findings |

Agent definitions are in `agents/` (Claude Code subagent format); `tools/install_agents.py` copies them to the workspace's `.claude/agents/`.
