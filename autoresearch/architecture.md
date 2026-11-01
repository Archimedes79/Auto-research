# autoresearch: two loops through four layers

![autoresearch: two loops through four layers](architecture.svg)

The person gives the prompt and decides; agents read and write files; code reads the files, runs them and writes its records back. The outer loop (problem, research, branches, tracking, promotion, cleanup) runs one round after another; the inner loop (build, train, evaluate, decide) runs once per experiment, several at a time. How to work in it: `METHOD.md`.

Why it looks like this (decided as D32 and finished in its lean form as D34, 2026-11-01; the migration record is `history/TARGET_2026-11-01.md`):
- **Files are the only interface.** A round can be resumed, reviewed in git and explained from its folder.
- **Few hard rules, in code; everything else is a default an agent may change** (`METHOD.md`, "Hard rules and defaults").
- **A role is a different context, not a different step.** Separate agents only for fresh eyes (critic) and parallel work (scans, lenses).
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
| Researcher scouts, summaries | agent role (mode) | `agents/ar-researcher.md`, `prompts/research_steps.md` | only on a trigger (METHOD.md, Planning a round): scouts in parallel, one per topic, then a summary with its digest per problem group |
| notes, summaries K register | plain file in the problem folder | `templates/paper_note.md`, `templates/group_summary.md` |  |
| the problem's literature tools | the problem's own code | `../bev_fusion_lane_center/tools/literature/catalogue.py` | no code in autoresearch/ yet; the lane problem's catalogue would move up once a second problem uses it |
| Planner: branches (one per lens); critic | agent role (mode) | `agents/ar-planner.md`, `agents/ar-critic.md` | the planner writes cards once per lens, in parallel; the critic (fresh context) accepts, revises or rejects each card |
| experiments/E<id>/ experiment.md (card block) | plain file in the problem folder | `templates/experiment.md`, `templates/plan.md`, `schemas/experiment.schema.json` | the round's plan (PLAN.md); each card names the version it starts from and its decision rule; the card block of experiment.md is what decide.py reads |
| experiment.py card.py, schemas | code: core and suites | `tools/experiment.py`, `tools/card.py`, `schemas/` |  |
| Planner, coder + critic | agent role (mode) | `agents/ar-coder.md`, `agents/ar-critic.md` | only for a block experiment (new code): detail plan naming the boxes it changes, code with its spec test, a critic pass on shared code; a variant needs no agent |
| detail plan, code tagged commit | plain file in the problem folder | `templates/experiment.md` |  |
| spec tests selftest.py | code: core and suites | `tools/selftest.py` |  |
| run records run.json, log.jsonl | plain file in the problem folder | `trainsuite/runs.py` | written by the launcher and the training; read by the train suite |
| START GATE card, slot, tag | transition point: refuses | `tools/run.py`, `tools/machine_lock.py` | run.py starts a run only with a card (hypothesis, prediction, rule), a free GPU slot, no other owner's exclusive machine lock (D19), the TEST split only with a logged row; it tags the code it ran |
| launcher + train suite, kill rules | code: core and suites | `tools/run.py`, `tools/schedule.py`, `tools/ledger.py`, `tools/plan.py`, `tools/run_report.py`, `tools/hold.py`, `trainsuite/` | a training is watched by code: the card's kill rules stop it (`watch --enforce`), `trainsuite early` only advises from the learning curves; an alarm wakes the lead chat, no agent watches |
| card.json, verdict (ledger derived) | plain file in the problem folder | `schemas/card.schema.json` |  |
| eval suite (VAL) decide.py: the rule | transition point: refuses | `evalsuite/`, `tools/decide.py`, `tools/provenance.py`, `tools/lineage.py` | numbers, pictures and ms on VAL; decide.py turns the cards into KEEP / CUT / TIE / INCONCLUSIVE by the card's written rule |
| Promotion: OK and TEST once | transition point: refuses | `tools/run.py` | the person's: a version becomes the deployed champion only with their OK and one TEST check, logged before it runs (run.py refuses otherwise) |
| Planner: tracking n best, next round | agent role (mode) | `agents/ar-planner.md` | reads the verdicts, updates the set of n best by the written rule, proposes a promotion, plans the next branches |
| the n best versions registry, champion | plain file in the problem folder | `../bev_fusion_lane_center/models/registry.json`, `../bev_fusion/records/champion.json` | the problem's list of good versions: lane models/registry.json (several named systems), bev_fusion records/champion.json (base, options); the tracking planner updates it, promotion stays the person's |
| board, prototype (per problem) | the problem's own code | `../bev_fusion/evaluation/board.py`, `../bev_fusion_lane_center/tools/prototype/lane_prototype.py`, `tools/keep3.py`, `../bev_fusion/records/keep3.json`, `../bev_fusion_lane_center/experiment_execution/keep3.json` | tools/keep3.py keeps the best three networks complete (one slot per checkpoint, per arm with its seeds in bev_fusion; the records are the keep3.json of each problem); the deployment suite: each problem's prototype command runs a version on chosen frames, writes planner-ready outputs and copies the numbers of its metric card and the ms per frame of its timing record (bev_fusion: evaluation/bev_prototype.py); a shared tool only when a third problem needs one |
| Critic numbers of the report | agent role (mode) | `agents/ar-critic.md` | a fresh reviewer checks the round's report against the primary sources |
| DECISIONS.md HANDOVER.md | plain file in the problem folder | `templates/plan.md` |  |
| cleanup, checks clean_temp, archdraw | transition point: refuses | `tools/clean_temp.py`, `tools/check_structure.py`, `tools/check_numbers.py`, `tools/metrics.py`, `tools/archdraw.py`, `tools/check_git.py`, `architecture.json` | tags the scratch and moves temp/ away, finds dead paths, checks the report's numbers, the per-round table, these drawings |

## Arrows

| from | to | carries |
|---|---|---|
| Planner: tracking n best, next round | Promotion: OK and TEST once | proposal |
| the n best versions registry, champion | detail plan, code tagged commit | start from a version |
| Planner: tracking n best, next round | Planner: branches (one per lens); critic | next round |
<!-- archdraw:end -->
