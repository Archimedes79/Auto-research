# The autoresearch method

A general workflow for improving a technical system by research, with agents doing most of the work and a person setting direction and granting permissions. It knows nothing about lane graphs; a problem plugs in through one folder with `PROBLEM.md` + `project.json` (see `README.md`).

## Two systems

| | the product | the improver (this method) |
|---|---|---|
| what | the system being improved (here: a lane-centre-line graph from camera + radar with SD-map assignment) | research -> hypotheses -> plans -> experiments -> decisions -> summaries |
| code | the problem's code folders (training, inference, data) | `autoresearch/tools/` (launcher, ledger, schedule, decisions, checks, cleanup) + the problem's evaluation code |
| changed by | reviewed commits; every experiment runs a tagged commit | improvements to this folder, proposed by the system-review lens and decided by the person |

## The cycle

```
 P0 FRAME            P1 RESEARCH (hierarchical)        P2 HYPOTHESES + PLAN              P3 LOOP                                P4 CLOSE
 PROBLEM.md          brief -> ar-scout: paper notes    ar-hypothesis (6 lenses): cards   variant: experiment file -> run ->     end of loop in the plan,
 metrics, protocol   ar-scout deep: port cards         ar-referee: accepted cards        evaluate -> decide -> summary line     report, card statuses,
 current best, gaps  ar-summarizer: group summary      ar-planner: plan + schedule       block: detail plan -> code + review    closed questions, cleanup,
 problem groups      with its digest                   system improvements               -> run -> evaluate -> decide -> summary  handover
```

A cycle is P1-P4. P1 and P2 are started by the person step by step (`prompts/research_steps.md`), before a loop or while a previous loop runs. P3 is the `/loop` (`prompts/loop.md`, which starts with the checklist for starting a loop). P4 closes it and feeds the next cycle. Each phase ends with a gate; a phase reads only the summaries of the phase before it and opens details only for the ids it uses. **Light cycle by default**: research only the groups whose gap or literature changed, three lenses; a full cycle every third time.

### P0 Frame (once per problem, revisited each cycle)
`PROBLEM.md` (`templates/problem.md`): goal and constraints, the product chain, the metrics and the evaluation protocol, the current best with its numbers, the known gaps, the **problem groups** (the taxonomy every later artifact is filed under) and the **foundation topics** (building blocks, training and losses, mathematical formulations). The person owns this file.

### P1 Research: from paper details to summaries
| step | who | writes |
|---|---|---|
| brief | lead | per scout: the question, what we know (ids), what to look for and to ignore, the budget, the family (state of the art / foundations and building blocks / training and losses) |
| scan | one `ar-scout` per topic, in parallel | **paper notes** (`templates/paper_note.md`): what, how, results, relevance per problem group (0-3), transferable technique, cost, code / licence, fact status (checked / digest / memory) |
| deep | `ar-scout` in deep mode for the best notes | an **extraction with a port card**: interface, what to take, licence, minimal spec test |
| summary | one `ar-summarizer` per group or topic | **group summary** (`templates/group_summary.md`) headed by its **digest** (the three impulses for the planners); ranked techniques with K ids, open questions, what not to do |

Gate G1: every group under attack has a summary; every technique named in it has a K id with a source and a fact status.

### P2 Hypotheses and plan
`ar-hypothesis` runs once per lens, in parallel, on the same inputs (the digests, the best metric card, the experiment summaries, the ledger, the closed questions). Each writes **experiment cards** (section 1 of `templates/experiment.md`).

| lens | question it asks |
|---|---|
| **state of the art** | what do the best methods on this problem do that we do not, and which of it fits our constraints? |
| **foundations** | which building block, structure or mathematical formulation would express the problem better? |
| **training and losses** | which training regime or loss would remove a known failure? |
| **past critic** | which past conclusions were drawn too early, confounded, missing controls, contradicted, or repeated negatives (close them)? |
| **system review** | where does the software foundation lack (speed, checks, reuse, tools, provenance)? |
| **free** | any idea the agent believes in, argued like the others |

Then `ar-referee` checks every card (accept / revise / reject with a reason), and `ar-planner` writes the **plan** (`templates/plan.md`: the tree of the levels this loop owns, system improvements, controls) and the **schedule** (one row per work item; the schedule tool keeps it; a disrupted loop resumes from it).

Gate G2 (hand-over ready, checked by the referee): a falsifiable statement, a mechanism, a prediction with a number and a metric, a decision rule with the noise band, controls (null, matched reference, equal budget, seed twin), a cost, a kill criterion, source ids, inputs ready or listed as tasks. The machine-readable part of the card (arms, seeds, metric, band, rule clauses) goes into the experiment file the tools read.

### P3 Loop: two kinds of experiment
- **Variant** (a configuration of existing code): experiment file -> the launcher runs every arm and seed (commit + tag per run) -> the evaluation entry point writes a metric card per run -> the decision tool applies the rule clauses to the cards (KEEP / CUT / TIE / INCONCLUSIVE; INCONCLUSIVE adds a seed) -> one summary line in the experiment file. No detail planner, no coder.
- **Block** (new code): `ar-planner` writes the detail plan (section 2 of `experiment.md`) -> `ar-coder` implements the change with its spec test; a change of shared code gets `ar-reviewer` (mode code) -> then as a variant -> summary (section 3).

Wake-up routine: read HANDOVER -> `tools/schedule.py show` and `tools/ledger.py tail` (the state of a run is read from its run record, nothing is synced) -> `tools/tick.py` starts ready items that have a command (a chain of `run.py --after` launchers does the same by hand) -> do what needs a chat (detail plan, code, decide, summary) -> ledger notes, plan status, HANDOVER -> next wake-up. Nothing else is state.

Gate G3 (per experiment): every number comes from a metric card or a run record; the rule was applied as written; the summary has open points and next steps.

### P4 Close
- End-of-loop section of the plan and the report from the experiment summaries; `tools/check_numbers.py` on the text; `ar-reviewer` (mode numbers) against the primary sources.
- Each card gets a status (confirmed / refuted / open / closed); a question answered the same way twice goes into the closed-questions file; open points go to the next cycle; decisions for the person go into the decisions file.
- Commit the experiment code (scratch included) so the tags keep it, then `tools/clean_temp.py loopNN --move`; `tools/check_structure.py`; rewrite HANDOVER (<= 3 KB).

Gate G4: tests pass, structure check clean, every run of the loop has a ledger decision and a summary, HANDOVER rewritten.

## Disruptions
A loop can be stopped at any moment (user, crash, shutdown, another chat).
- **State in a few files**: HANDOVER (narrative, <= 3 KB: where we are, what runs, next step, the lead chat that owns the code, pointers), the plan, the schedule, the ledger, the decisions file. A new chat continues from these alone.
- **Reading**: single-file records (ledger, decisions, closed questions, plan) are complete files that agents search; experiment folders are read in full for the last two loops, older loops through their summaries and the problem's current best.
- **Idempotent steps**: every schedule item can be restarted; trainings checkpoint and resume; a done flag means exit code 0.
- **Read the records first**: after a disruption `schedule.py show` and `ledger.py tail` give done / failed / stalled / running from the run records (`ledger.py sync` adds rows of finished runs that have none); then the lead continues. A change of direction goes through `ar-planner` (a new schedule version), not into ad-hoc runs.
- **STOP_AT is a work window**: when the schedule empties early, `ar-planner` adds the next items (controls, seeds, robustness, backlog) with rules written before the runs.

## Decisions of the person
The person decides through one decisions file, never through file edits by agents: promoting a champion (which includes the one TEST check of the candidate and the current champion; TEST is never used otherwise), changes of `PROBLEM.md`, changes of the gates of this method (the protected files and the hook; other tool changes need a passing selftest and an ar-reviewer pass, not a decision), and permissions (downloads, installs, mail, shutdown, backups, remote). An agent writes the entry with options, recommendation and evidence, and acts only after the answer is recorded; an answer given in chat is recorded by the agent with the date and the person's words.

## Code and git
- The repository holds code and text records: this folder, the problem's code folders and scratch scripts (so a tag keeps them), and the text records (plan, schedule, ledger, decisions, run records, experiment files, metric cards, literature summaries). Never data, checkpoints, caches or binaries.
- Git history is the long-term memory; nothing committed is rolled away. Only what an agent reads rolls (see "Reading").
- `main` holds the reviewed core bundle; one lead chat per loop owns the code; other chats work in research or reports or only read.
- Every run commits the current code and tags it `<prefix>/E<id>.<arm>`; the ledger and the metric card record the commit.

## Keeping the system small
A new template or agent is added only when the same kind of artifact has been produced twice by hand; until then it is a section of an existing file or a sentence in a brief. The same holds for tools: a script written for one loop stays in the loop's scratch; it becomes a tool when it is needed a second time.

## Roles
| agent | phase | input -> output |
|---|---|---|
| lead (main chat) | all | briefs, wake-ups, the decisions with the person |
| `ar-scout` | P1 | brief -> paper notes; deep mode: extraction + port card |
| `ar-summarizer` | P1 | notes of a group -> group summary with digest |
| `ar-hypothesis` (lens in the brief) | P2 | digests + past + best card -> experiment cards |
| `ar-referee` | P2 | cards -> verdicts, hand-over readiness |
| `ar-planner` | P2 / P3 | accepted cards -> plan + schedule; card -> detail plan; re-plan after a disruption |
| `ar-coder` | P3 | detail plan -> code + spec test |
| `ar-reviewer` | P3 / P4 | mode code: diff of shared code -> findings; mode numbers: report + primary sources -> findings |

Templates: `problem`, `paper_note`, `group_summary`, `experiment`, `plan` (`templates/`). Agent definitions are in `agents/` (Claude Code subagent format); `tools/install_agents.py` copies them to the workspace's `.claude/agents/`.
