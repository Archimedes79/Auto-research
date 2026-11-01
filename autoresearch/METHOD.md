# The autoresearch method

Improving a technical system by research: agents do most of the work, a person sets the direction and decides. A problem plugs in through one folder with `PROBLEM.md` + `project.json` (`README.md`). The picture of the system is `architecture.svg`; where it is going is `TARGET.md`.

**Four layers.** The **person** gives the prompt and decides. **Agents** (four roles, below) read files and write files. **Files** in the problem folder are the only contract between agents and code: problem, research notes, experiment cards, plan, run records, metric cards, ledger, decisions, handover. **Code** (`tools/`, `trainsuite/`, `evalsuite/`) runs, measures, refuses and applies rules written before the run; judgment stays with agents and the person. A phase reads only the summaries of the phase before it and opens details only for the ids it uses.

## Outer loop: one round per prompt
| phase | what happens | it ends when (gate) |
|---|---|---|
| P0 Problem | the planner (mode problem) drafts changes of `PROBLEM.md`; the person approves them | the person's answer is recorded |
| P1 Research | researchers in parallel, one scan per topic, deep mode for the best papers, then one summary with its digest per problem group (`prompts/research_steps.md`) | G1: every group under attack has a summary; every technique named has a K id with a source |
| P2 Branches | the planner writes cards once per lens, in parallel; the critic (mode cards) returns accept / revise / reject per card and the lead saves that report as a file next to the plan; the planner (mode plan) writes the verdicts into the cards, then the plan, the schedule, the planned ledger rows and the `experiment.json` of every card the tools decide | G2: every card is hand-over ready (falsifiable, a mechanism, a prediction with a number, a rule with the noise band, controls, cost, kill criterion, source ids, inputs); a card the tools decide has its `experiment.json` (`experiment.py check`) |
| P3 Loop | the inner loop for every card (`prompts/loop.md`) | G3 per card: every number from a metric card or run record, the rule applied as written, a summary with next steps |
| P4 Close | the planner (mode tracking) reads the verdicts, lists which versions are now among the best (today in the problem's registry: lane `models/registry.json`, bev_fusion `records/champion.json`; a `versions/` folder is a later step, `TARGET.md`), proposes a promotion; the critic (mode numbers) checks the report; a question answered the same way twice goes into the closed-questions file; scratch is tagged and moved away | G4: tests pass, structure check clean, every run has a ledger decision and a summary, HANDOVER rewritten |

Rounds run one after another: a round ends with decisions the next one needs. Several versions can be good at once (fast, accurate); a card names the version it starts from, so one round branches from several of them.

## Inner loop: one per card, several at a time
- **Variant** (existing code, other settings): each arm and seed has a planned ledger row with the card's hypothesis, prediction and rule; the lead starts it through `run.py` (a chain with `--after`; commit + tag per run) -> the evaluation writes a metric card per run -> `decide.py` applies the rule of `experiment.json` (KEEP / CUT / TIE / INCONCLUSIVE; INCONCLUSIVE adds a seed) -> one summary line. No agent judges.
- **Block** (new code): the planner (mode detail) writes the detail plan, naming the boxes of the system drawing it changes -> the coder implements it with its spec test -> the critic (mode code) reviews shared code -> then as a variant.
- Trainings are watched by code where the trainer follows the `trainsuite` contract (today bev_fusion): the lead writes the card's kill criterion as a rule into the trainer's run folder (`<run>/kill.txt`, or `--kill`) and starts `trainsuite watch <runs> --enforce` as a Monitor, which stops a run that hits it; an alarm wakes the lead, no agent watches. For other trainers (the lane problem today) the lead applies the kill criterion.

## The rules that are code
A capable agent can argue its way around a rule in prose, so the rules that protect the truth refuse in code: the **write gate** (`guard_hook.py`: protected files only with a grant under a decided D<n>), the **start gate** (`run.py`: no run without a card; the TEST split only with a logged row; a tagged snapshot), the **decision rule** (`decide.py`), **promotion** (the person's, with the one TEST check; TEST is never used otherwise) and the **cleanup** (`clean_temp.py` refuses to delete untagged scratch, `check_structure.py` reports dead paths; `check_numbers.py` lists numbers without a source and refuses only with `--strict`). Everything else is advice in a template.

## Disruptions
A loop can stop at any moment. The state is a few files (HANDOVER <= 3 KB, plan, schedule, ledger, decisions); a new chat continues from them alone. Every step is restartable: trainings checkpoint and resume, a done flag means exit code 0. After a disruption read the run records first (`schedule.py show`, `ledger.py tail`); a change of direction is a new schedule version from the planner, never ad-hoc runs. STOP_AT is a work window: when the queue empties early, the planner adds controls, seeds and robustness with rules written before the runs. Single-file records (ledger, decisions, closed questions, plan) are searched whole; experiment folders of the last two loops are read in full, older ones through their summaries.

## Decisions of the person
Through one decisions file, never through edits by agents: promoting a champion (with its one TEST check), changes of `PROBLEM.md`, changes of the gates (protected files, the hook), permissions (downloads, installs, mail, shutdown, backups, remote). An agent writes the entry with options, recommendation and evidence, and acts only on the recorded answer; an answer given in chat is recorded by the agent with the date and the person's words.

## Code, git, size
The repository holds code and text records, never data, checkpoints or binaries; git history is the long-term memory. One lead chat per loop owns the code; a change of shared code gets a critic pass, a change of a tool also a passing `tools/selftest.py`. A template, agent or tool is added only when the same thing was done twice by hand, and removed when nothing uses it.

## Roles
| role | modes | phase |
|---|---|---|
| lead (the main chat) | briefs, wake-ups, decisions with the person | all |
| `ar-planner` | problem, branches (one lens per run), plan, detail, tracking | P0, P2, P3, P4 |
| `ar-researcher` | scan, deep, summary | P1 |
| `ar-coder` | | P3 |
| `ar-critic` | cards, code, numbers | P2, P3, P4 |

Templates: `problem`, `paper_note`, `group_summary`, `experiment`, `plan`. `tools/install_agents.py` copies `agents/` to the workspace's `.claude/agents/`; the old names (`ar-scout`, `ar-summarizer`, `ar-hypothesis`, `ar-referee`, `ar-reviewer`) forward to the new roles until the loop running at the switch has ended.
