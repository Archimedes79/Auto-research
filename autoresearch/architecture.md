# autoresearch: two loops through four layers

![autoresearch: two loops through four layers](architecture.svg)

The person gives the prompt and decides; agents read and write files; code reads the files, runs them and writes its records back. The outer loop (problem, research, branches, tracking, promotion, cleanup) runs one round after another; the inner loop (build, train, evaluate, decide) runs once per experiment, several at a time. How to work in it: `METHOD.md`.

Why it looks like this (decided as D32 and finished in its lean form as D34, 2026-11-01; the migration record is `history/TARGET_2026-11-01.md`):
- **Files are the only interface.** A round can be resumed, reviewed in git and explained from its folder.
- **Few hard rules, in code; everything else is a default an agent may change** (`METHOD.md`, "Hard rules and defaults").
- **A role is a different context, not a different step.** Separate agents only for fresh eyes (critic) and parallel work (scans, lenses).
- **Training is general, the network is not (D48).** The loop of a training (checkpoints with resume, EMA, clipping, the stop request) is `trainkit/`, the same for every problem; what is trained (model, loss, data) stays in the problem's `training/`. The Training and `run.py` see only the files the loop writes.
- **The n best versions are what a round updates;** a card names the version it starts from, so a round can branch from several.
- **Build on the second use, remove what is unused.** What was deliberately not built: a shared deployment tool (each problem's prototype command does it), a `versions/` folder (the registry and champion files hold the versions), layer folders in `autoresearch/` (the drawing shows the layers), a derived schedule (`schedule.csv` is an optional to-do list whose states come from the run records).

<!-- archdraw:begin: generated from architecture.json by autoresearch/tools/archdraw.py; edit the spec, then `archdraw.py render` -->
## File map: every box and the code it stands for

| box | kind | code | note |
|---|---|---|---|
| Prompt; approves the problem | the person | `prompts/loop.md` | the person starts a round with a prompt (the filled loop prompt: stop time, permissions) and approves the problem the planner drafted |
| Planner problem draft | agent role (mode) | `agents/ar-planner.md` | the planner in mode problem drafts changes of PROBLEM.md from the prompt; the file stays protected, the person approves |
| WRITE GATE: protected truth files only with a grant under a decided D<n>; a card only when complete (G2) | transition point: refuses | `tools/guard_hook.py`, `tools/experiment.py` | transition agents -> files |
| PROBLEM.md project.json | plain file in the problem folder | `templates/problem.md` |  |
| project.py install_agents.py | code: core and suites | `tools/project.py`, `tools/install_agents.py`, `config.json` |  |
| Researcher scan, deep, check | agent role (mode) | `agents/ar-researcher.md`, `prompts/research_steps.md` | on a trigger or the person's order (METHOD.md, Planning a round; research_steps.md step 1 is the contract): scans in parallel make the candidate list, one deep agent per candidate writes a note, a fresh agent checks a sample against the sources, one summary agent writes the overview |
| topic notes, table overview, K register | plain file in the problem folder | `templates/paper_note.md`, `templates/group_summary.md` |  |
| topic_table.py + lane catalogue | the problem's own code | `tools/topic_table.py`, `../bev_fusion_lane_center/tools/literature/catalogue.py` | topic_table.py is shared by every problem (the table of a topic, made from the notes' header lines; --set-result fills the experiment result of a paper); catalogue.py is the lane problem's older index of papers.csv |
| Planner: branches (one per lens); critic | agent role (mode) | `agents/ar-planner.md`, `agents/ar-critic.md` | the planner writes cards once per lens, in parallel; the critic (fresh context) accepts, revises or rejects each card |
| experiments/E<id>/ experiment.md (card block) | plain file in the problem folder | `templates/experiment.md`, `templates/plan.md`, `schemas/experiment.schema.json` | the round's plan (PLAN.md); each card names the version it starts from and its decision rule; the card block of experiment.md is what decide.py reads |
| experiment.py card.py, schemas | code: core and suites | `tools/experiment.py`, `tools/card.py`, `schemas/` |  |
| Planner, coder + critic | agent role (mode) | `agents/ar-coder.md`, `agents/ar-critic.md`, `prompts/experiment_flow.md` | only for a block experiment (new code): detail plan naming the boxes it changes, code with its spec test, a critic pass on shared code; a variant needs no agent; prompts/experiment_flow.md is one experiment in six steps (plan card, coding, experiment, training, evaluation, store to the result folder) |
| detail plan, code tagged commit | plain file in the problem folder | `templates/experiment.md` |  |
| spec tests selftest.py | code: core and suites | `tools/selftest.py`, `tools/selftest_verdict.py`, `tools/selftest_launch.py` | the tools' own tests: selftest.py runs every tool's selftest and the two lifecycle tests (selftest_launch.py: the launcher, the TEST log and the whole chain run -> decide -> result with the real tools; selftest_verdict.py: the verdict path), in a throw-away problem |
| run records run.json, log.jsonl | plain file in the problem folder | `Training/runs.py` | written by the launcher and the training; read by the train suite |
| START GATE card, slot, tag | transition point: refuses | `tools/run.py`, `tools/machine_lock.py`, `tools/testgate.py` | run.py starts a run only with a card (hypothesis, prediction, rule), a free GPU slot, no other owner's exclusive machine lock (D19), the TEST split only with a logged row under the person's OK or a decided grant (testgate.py: the one rule, also loaded by the lane's evaluator); it stores the hash of the card's rule in the run record (the gate of decide.py compares it) and tags the code it ran |
| launcher + train suite, kill rules | code: core and suites | `tools/run.py`, `tools/schedule.py`, `tools/ledger.py`, `tools/plan.py`, `tools/run_report.py`, `tools/hold.py`, `Training/` | a training is watched by code: the card's kill rules stop it (`watch --enforce`), `Training early` only advises from the learning curves; an alarm wakes the lead chat, no agent watches; `Training start` composes and launches, `Training queue` shows the GPU slots and the line of waiting runs; the loop that is watched is the engine below |
| training engine trainkit: fit loop | code: core and suites | `trainkit/` | D48: the project-independent training loop (accumulation, EMA, learning-rate schedule per module, clipping, non-finite skip, AMP, atomic checkpoints with resume, the stop request); a project supplies model, loss, data and config (contract in trainkit/README.md); it writes log.jsonl, checkpoints and STOPPED, reads CONTROL, so the Training and run.py need no project code; the BEV trainer bev_fusion/training/train.py is its first user, a run trains from a pinned copy in runs/<id>/code |
| card.json, verdict (ledger derived) | plain file in the problem folder | `schemas/card.schema.json` |  |
| eval suite (VAL) decide.py: the rule | transition point: refuses | `Evaluation/`, `tools/decide.py`, `tools/result.py`, `tools/provenance.py`, `tools/lineage.py`, `tools/package.py` | numbers, pictures and ms on VAL; decide.py turns the cards into KEEP / CUT / TIE / INCONCLUSIVE by the card's written rule and refuses a run that has not finished, was started without a card or is being repeated, and a rule edited after the launch (the hash run.py stored); result.py then writes results.md, the schedule link and the research notes' result line; package.py writes the result package of a run (with the scene lists in split.json) and cleans derived data |
| Promotion: OK and TEST once | transition point: refuses | `tools/run.py` | the person's: a version becomes the deployed champion only with their OK and one TEST check, logged before it runs (run.py refuses otherwise) |
| Planner: tracking n best, next round | agent role (mode) | `agents/ar-planner.md` | reads the verdicts, updates the set of n best by the written rule, proposes a promotion, plans the next branches |
| the n best versions registry, champion | plain file in the problem folder | `../bev_fusion_lane_center/models/registry.json`, `../bev_fusion/records/champion.json` | the problem's list of good versions: lane models/registry.json (several named systems), bev_fusion records/champion.json (base, options); the tracking planner updates it, promotion stays the person's |
| board, prototype (per problem) | the problem's own code | `../bev_fusion/evaluation/board.py`, `../bev_fusion_lane_center/tools/prototype/lane_prototype.py`, `tools/keep3.py`, `../bev_fusion/records/keep3.json`, `../bev_fusion_lane_center/experiment_execution/keep3.json` | tools/keep3.py keeps the best three networks complete (one slot per checkpoint, per arm with its seeds in bev_fusion; the records are the keep3.json of each problem); the deployment suite: each problem's prototype command runs a version on chosen frames, writes planner-ready outputs and copies the numbers of its metric card and the ms per frame of its timing record (bev_fusion: evaluation/bev_prototype.py); a shared tool only when a third problem needs one |
| Critic numbers of the report | agent role (mode) | `agents/ar-critic.md` | a fresh reviewer checks the round's report against the primary sources |
| decisions/ results.md HANDOVER.md | plain file in the problem folder | `templates/plan.md`, `templates/results.md`, `tools/result.py` | decisions/D<n>.md is the only channel for the person's answers (the gate reads its Decided line); results.md is written by result.py after every verdict (generated block between markers, the author's text below); HANDOVER.md is rewritten at every milestone |
| cleanup, checks clean_temp, archdraw | transition point: refuses | `tools/clean_temp.py`, `tools/check_structure.py`, `tools/check_numbers.py`, `tools/metrics.py`, `tools/archdraw.py`, `tools/check_git.py`, `tools/commit_paths.py`, `architecture.json` | tags the scratch and moves temp/ away, finds dead paths, checks the report's numbers, the per-round table, these drawings; commit_paths.py commits exactly the named paths of a unit of work (scratch and forbidden files refused) |

## Arrows

| from | to | carries |
|---|---|---|
| Planner: tracking n best, next round | Promotion: OK and TEST once | proposal |
| the n best versions registry, champion | detail plan, code tagged commit | start from a version |
| Planner: tracking n best, next round | Planner: branches (one per lens); critic | next round |
| launcher + train suite, kill rules | training engine trainkit: fit loop | CONTROL / STOPPED, log.jsonl |
<!-- archdraw:end -->
