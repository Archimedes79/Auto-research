# autoresearch - a research system around a product

General and problem-independent: the method (`METHOD.md`), four agent roles (`agents/`), six templates (`templates/`), three prompts (`prompts/`) and the tools that make the method hold (`tools/`). A problem plugs in through its own folder. It is kept small on purpose: a new template, agent or tool only when the same thing has been done twice by hand (`METHOD.md`, "Code, git, size"), and only a few rules are hard (`METHOD.md`, "Hard rules and defaults").

## Start here: the whole system on one page

A problem is a folder. The person writes `PROBLEM.md` (goal, metric, splits, current best); everything else is written by agents and tools.

| file (in the problem folder) | what it holds | who writes it |
|---|---|---|
| `PROBLEM.md` | the problem; protected (changes only under a decided D<n>) | the person; the planner drafts |
| `experiments/<id>/experiment.md` | one file per experiment: the card (question, prediction as a number, decision rule, kill criterion) as a `json card` block, later the summary | the planner or the lead |
| `schedule.csv` (optional) | the list of experiments as work items: `node` = the experiment, `depends_on`, `resource`, `run_ids`, `command`; the state is derived from the run records, never stored; `plan.py status` lists the experiments with their runs and verdicts | the planner or the lead |
| `LEDGER.csv` | the run log, one row per run: written by `run.py` and `decide.py`, edited by nobody | the tools |
| `decisions/` (one file D<n>.md each), `HANDOVER.md` | the questions to the person and the recorded answers; where we stand (<= 3 KB) | agents write, the person answers |

One experiment: write the card -> `run.py` (refuses without a complete card, tags the code, writes the run record) -> the evaluation writes the metric card -> `decide.py` applies the rule of the card (KEEP / CUT / TIE / INCONCLUSIVE) -> the summary goes into the experiment file. Six rules are hard (`METHOD.md`, "Hard rules and defaults"); everything else is a default. The rest of this page is the reference: tools, suites, drawings.

![autoresearch overview: you, one lead agent, a notebook of plain files, experiment agents, code that measures and refuses](overview.svg)

The overview above is generated from `overview.json`; the detail with every box and every tool is [architecture.md](architecture.md) (the picture below, 27 boxes).

![autoresearch: two loops through four layers. The person decides, agents write plain files, code runs them and writes the records back](architecture.svg)

The architecture in one picture: four layers - the **person**, the **agents** (four roles), **plain files in the problem folder** and **code** (the tools and the two suites) - and two loops: the outer one (problem, research, branches, tracking of the n best versions, promotion, cleanup) runs one round after another, the inner one (build, train, evaluate, decide) once per experiment, several at a time. Nothing but files passes between agents and code, so every step can be read, diffed in git and resumed. The points that can refuse are drawn red: the write gate (the gate hook and the card check), the start gate (`run.py`: a card, a logged TEST row, a GPU slot, a tagged snapshot), the decision rule (`decide.py`), the promotion (the person, with one TEST check) and the cleanup (`clean_temp.py` tags the scratch and moves it away). [Each box with the code it stands for](architecture.md). The drawings are generated from `architecture.json` (and the two problems' `training/architecture.json`) by `tools/archdraw.py`, which also fails when a box names a file or symbol that does not exist, or a code file is in no box (section "Examples").

| folder / file | content |
|---|---|
| `METHOD.md` | one page: the outer loop P0 problem -> P1 research -> P2 branches -> P3 inner loop -> P4 close, the inner loop (variant and block experiments), the hard rules and the defaults an agent may change, disruptions, decisions of the person, code and git, roles |
| `suitecli.py` | the one command-line generator of the suites: every method of a suite's class is a command (`python3 suitecli.py --selftest`); the convention is the section 'The interface of the suites' below |
| `agents/` | four roles (D32): `ar-planner` (modes problem, branches with six lenses, plan, detail, tracking), `ar-researcher` (scan, deep, summary), `ar-coder`, `ar-critic` (cards, code, numbers: fresh eyes) - Claude Code subagent format (the forwarding files of the old role names were removed on 2026-11-01, when the loop running at the switch had ended) |
| `templates/` | `problem`, `paper_note`, `group_summary` (with its digest), `experiment` (card block -> detail plan -> summary, all in one file), `plan` (tree + end of loop) |
| `prompts/` | `research_steps.md` (step 0 start of a round, P1 on a trigger, P2 in steps started by hand), `loop.md` (P3 + P4, pasted after `/loop`), `experiment_flow.md` (one experiment in six steps: plan card, coding, experiment, training, evaluation, store to the result folder) |
| `tools/` | `run.py` (launcher: ledger row, commit + tag per run with a code-drift check at the end, TEST guard, retries), `ledger.py`, `schedule.py`, `plan.py` (status of the plan nodes from the schedule and the ledger), `commit_paths.py` (commit exactly the named paths in the shared checkout: temporary index, compare-and-swap on HEAD, scratch and forbidden files refused), `decide.py` (the written rule as code), `check_numbers.py`, `check_structure.py`, `check_git.py` (what may go into git and be pushed: refuses staged scratch (`temp/`, `_to_delete/`) and forbidden files, unpushed commits that add them, lists the run tags that are not on the remote and the scratch that living files point at; rule: root `CLAUDE.md`), `keep3.py` (the best three networks of a problem: one slot per checkpoint, a text record `keep3.json` next to the ledger, a candidate comes in by more than the band with a ledger KEEP or "by hand: <reason>", checkpoints of the three stay complete; the push on a change is a standing instruction, root `CLAUDE.md`), `clean_temp.py`, `install_agents.py` (copies the agent definitions to `.claude/agents/` after a lint: frontmatter, English ASCII text, no paths, names, dates or run ids, existing references; `--check`, `--selftest`), `guard_hook.py` (gate hook: Write / Edit of protected files only with a grant under a decided D<n>), `machine_lock.py` (D19: one machine lock between chats; `run.py` waits while another owner holds an exclusive one), `metrics.py` (per-loop table: runs, GPU hours, verdicts, prediction hits, TEST uses), `run_report.py` (one page per training run for a person without code: state, resumes, the card's prediction and rule, curves, the numbers next to reference runs, the compute in TOPS, pictures and videos each with a review field that exports a `review.json`; `--ingest` files it; `--selftest`; BEV: `bev_suite.py RUN report`), `project.py` (finds the problem and its `project.json`), `selftest.py` |
| `tools/provenance.py` | the provenance block of a result package (GT version and mode, evaluator code hash, frames, model, environment), `compare` of two cards, `package.py` (the result package of a run in `<problem>/results/<run id>/` and the clean-up of derived data and old GT stores, D40; `project.json` `package.split_record` = a command that writes `results/<id>/split.json`, a package is refused without it), `manifest <run id>` (the provenance of a training run from its record and run folder, with the model bundle: the network and loss modules by import closure, hash and copy), `repro <run id>`; design `reproducibility.md` |
| `tools/lineage.py` | the chain network -> training run -> code -> data -> evaluations, every link verified (`show <name / run id / checkpoint>`, `verify [--all]` over the keep3 networks: checkpoint size + hash, run tag -> commit -> ancestor / tag, code drift, autogt version + tag, card provenance and manifest hash; OK / MISSING / MISMATCH, exit 1 on a MISMATCH), `record <run id> --ckpt --parent` writes `lineage.json` (the parent checkpoint of future networks); read-only otherwise, selftest |
| `tools/topic_table.py` | the table of a research topic, made from its notes and never typed: one subfolder per paper with a `note.md` whose eight header lines (title, relevance 0-3, one line of about three sentences, status, paper link, code link, verdict, experiment result) become `table.md` and `table.csv`, best relevance first; a bad note, a folder without a note, an empty topic or two notes on one paper make it print the reasons, remove any older table (also table_min*) and exit 1. Options: `--check` (the done state of a topic: fresh table, links, check.md without a failed claim, every candidate with a folder), `--known <papers.csv>` (marks papers the problem already has), `--set-result <note id> "<exp id>: KEEP ..."` (+ `--all`: every copy of the paper across the topics), `--min-relevance N`; a slug is looked for in the research folder of the problem named by AR_PROJECT (`research_steps.md` step 1; the planner reads the table) |
| `evalsuite/` | the evaluation suite for any network: pictures (one frame, random frames, result videos), numbers (over all videos, outliers traced back to frames, paired comparison) and cost (ms and GMACs per frame, the Pareto front of quality against cost) behind three small boxes (Source / Subject / Task); one-page `README.md` with a 70-line minimal adapter, `EXTENSIONS.md` for memory, taps and probing; `stats.py` = the one paired scene bootstrap of the repository (variant - control, also used by the lane project); `comparison/` = the one reviewed place where a prediction is compared with the ground truth (pixel maps, box lists); `python -m evalsuite selftest`; adapters per problem in `bev_fusion/evaluation/` and `bev_fusion_lane_center/evaluation/` (`lane_suite.py`, the lane Task `lane_task/`, moved out of here by D24); numpy + matplotlib (scipy for the full selftest), not standard-library only |
| `trainsuite/` | see, stop and judge a training run from the files it writes (contract: `log.jsonl` metrics stream, `train.log`, `.done` only for exit 0, `CONTROL` / `STOPPED` with exit code 75): `status` table with ETA, `health` checks (crash, stall, NaN, spike, rising loss, slowdown, GPU, over-fit, comparison with a reference run), kill rules of the card (`group.lines@1000 > 1.48`), `curves`, `stop`, `watch` (one line per change, for a Monitor; `--enforce` stops on a kill hit), `early` (advisory early-stop advice from the gap to a control run, calibrated on finished pairs; it stops nothing), `compose` / `start` (a training config from a base file + presets + `--set`, resolved split, static check, the exact `run.py` command; `--go` launches it), `queue` (the 2 GPU slots, the first-come-first-served line, the RAM gate, the holds); `Training` (`api.py`: the one class with `check` / `run` / `status` / `list` / `stop` / `resume` / `plot` / `compare`, every method also a command, a job is a json file); standard library (curves: matplotlib); the one-page `README.md` (config and manifest, tie to data and code, start, stop, supervise, deep debugging, queue, speed-up chances; the training loop it watches is `trainkit/`), `EXTENSIONS.md` (everything else); `python -m trainsuite selftest`; the training of `bev_fusion/` is the first user (D23) |
| `runsuite/` | commented out, not deleted (Mirko 2026-11-10): the runner is not a suite of its own but ONE class `Runner` in `bev_fusion/runner/` with `python -m runner` above it, parallel to `Training` and `Evaluation`; the job layer that was here (a tracked start and a queue item around a runner command) stays in the files as comments |
| `trainkit/` | the project-independent training engine (D48): the fit loop (`loop.py`: resume or warm start, accumulation, the learning-rate schedule with an own rate per module, gradient clipping, the optimiser step with the non-finite rules and fp16 GradScaler, EMA, atomic checkpoints, the evaluation, the stop request) with hooks for probes, and the files around it (`log.jsonl` rules, checkpoints, the stop protocol `CONTROL` / `STOPPED` with exit code 75, which `trainsuite/control.py` re-exports); plain torch, no project code; a project supplies model, loss, data, config and the step function (contract and a minimal example in its `README.md`, `python -m trainkit.selftest`); the BEV trainer `bev_fusion/training/train.py` is its first user, guarded by `bev_fusion/tests/test_trainkit_equivalence.py` (the trainer from before the move against the live one: same losses, weights, files and messages, eight mutants killed) |
| `history/` | design history, not read by the tools: the long roadmap and reviews of 2026-10-25, the D22 tool-fix spec (applied), the evalsuite architecture, the repository reviews of 2026-10-31, the target design of D32 and its migration record (`TARGET_2026-11-01.md`, finished in its lean form by D34) |
| `reproducibility.md` | what a result package records (GT version and mode, evaluator code, frame list, model, environment), the gaps today and the plan: a `provenance` block and `repro.py` (design, not built) |
| `coordination.md` | design of two small mechanisms between chats: the hold marker for stopped runs and claims / freezes with a reason and a release (the hold marker is built: `tools/hold.py`, `run.py` refuses a held id, state `held`; claims are design only) |
| `fewer_files.md` | built 2026-11-01: the card is a block inside `experiment.md` (`experiment.py migrate`), the status of plan nodes comes from `plan.py status`; what was left out and why |
| `overview.json` | the overview drawing (8 boxes: you, the lead agent, the notebook, experiment agents, measuring code and gates, research and critic on demand), rendered to `overview.svg` / `overview.md` by `tools/archdraw.py`; the 27-box detail is `architecture.json` |
| `ROADMAP.md` | the open items of the system (backups, CONFIRM split, evaluation cache key, ...), what is done, and what the last reviews taught |
| `architecture.json` / `.svg` / `.md` | the drawing of this system and why it looks like this: the spec (boxes, arrows, the code each box stands for), the generated picture and the generated file map; `python tools/archdraw.py check architecture.json` keeps picture and code together |
| `config.json` | the default problem folder |

## Plug in a problem
1. Make a folder with `PROBLEM.md` (from `templates/problem.md`) and `project.json` (paths: ledger, schedule, runs, holdout log and `holdout_split`, code folders, tag prefix, scratch folder, structure rules, `headline_metrics`, `cards`, `experiment_dirs` = where the experiment folders live, default `["experiments"]`). Give it one evaluator that writes a metric card per named system and split as in `schemas/card.schema.json` (an ML model on a validation split, or a code system benchmarked by a script: the loop is the same).
2. Set it as default in `config.json` or export `AR_PROJECT=<folder>` for the tools.
3. Set a git identity in the repository's own config (both the Windows and the WSL git use it): `git config user.name ...`, `git config user.email ...`.
4. `python tools/install_agents.py` copies the agents into the workspace's `.claude/agents/` (`--check` tells when the copies drift).
5. List the protected files in `project.json` ("protected") and put the gate hook into the workspace's `.claude/settings.json` (PreToolUse, matcher `Write|Edit|MultiEdit|NotebookEdit`, command `python3 autoresearch/tools/guard_hook.py` run from the workspace; on Windows through `wsl.exe --cd "$CLAUDE_PROJECT_DIR"`). Claude Code reads that file only for sessions started in the workspace folder.
6. Run the research steps by hand (`prompts/research_steps.md`), then loops (`prompts/loop.md`).

Problems using it: `../bev_fusion_lane_center/` (lane-centre-line graph from camera + radar, assignment to an ADASIS-like SD map; tag prefix `lcr`; the default problem, the only one under the gate hook) and `../bev_fusion/` (camera + radar BEV system; tag prefix `bf`, since D23; tools with `AR_PROJECT=../bev_fusion`; its `gpu_slots_dir` points at the lane problem's slot folder, so both count against the 2 slots of the one GPU).

## The interface of the suites
One pattern for `autogt`, `trainsuite` (class `Training`), `runner` (class `Runner`, in `bev_fusion/runner/`) and `evalsuite` (class `Evaluation`): a class per suite, every
method also a command, generated by `suitecli.py`. The README of each suite lists its methods and its job file (`trainsuite/README.md` section 0, `bev_fusion/runner/README.md`, `evalsuite/README.md`);
`python -m <suite> methods` prints them.

1. **One class per suite, its methods are the interface.** `Training(project)`, `Runner(project)`; the project folder is the only state (default `AR_PROJECT`). Method names are snake_case,
   verb first (`check`, `run`, `status`). (autogt has the class `AutoGT` and the same calls as module functions: `autogt.generate_bev_gt`, `autogt.gt_key`.)
2. **Plain data in and out.** A function takes paths, ids and dicts and returns a dict or a list (json-serialisable: numbers, text, lists, dicts). It never prints. The command line prints.
3. **One error per suite** (`TrainingError`, `EvaluationError`, ...) whose message says what to fix; every method turns the engine's own error into it. The command line prints it to
   stderr and exits with 2.
4. **A job is a json file or the same dict.** It says what to do and with what: `id`, `kind`, `config`, `preset`, `set`, `card`, `after`, `priority`, `kill` ... An unknown key is an error
   (a typo shows). Keys that start with `_` are comments.
5. **Every method is ONE command, the same call as in Python, with the class name in front.** `python -m trainsuite Training.run job.json` is `Training().run("job.json")`
   (evalsuite: `python -m evalsuite Evaluation.run ...`). Short forms exist where no older command has the word (`run job.json`), and the hyphenated form of a longer name (`qual-frame` for `qual_frame`; autogt and evalsuite hyphenate their multi-word commands too). A parameter without a default is positional (except `words`), so is `i` and what a suite lists as its main things (`NETWORK FRAMES`); every other one is an option
   (`--wait`, `--kill` and `--taps` repeatable or comma separated, `--no-start` for a switch that is on by default); `--project` names the project folder. `python -m trainsuite methods` lists the methods with their arguments; the generator is `suitecli.py`. Every command takes `--json` and then prints exactly what the method returns (NaN and Infinity become null), otherwise a short text (the older words keep their own
   output). Exit codes: 0 done, 1 `check` found problems, 2 an error (a message on stderr, no traceback; `TRAINSUITE_TRACEBACK=1` shows it).
6. **The same verbs everywhere.** The core every suite has: `check` looks and changes nothing (a preview file at most), `run` / `generate` / `predict` does the work now, `status` reads
   one thing, `stop` asks one run to checkpoint and stop, `selftest` checks the suite on synthetic data. None of the suites has a queue: trainsuite keeps one as a mixin (`queue_api.py`) that is not part of `Training`; a suite that needs a queue adds it.
7. **`i` addresses a run.** An id (`t23.C_s1`) or a run folder name (`t23_C_s1`). Without `i`, `status` and `stop` mean the running training, also one that was started by hand (it holds a
   GPU slot of this project). A typo in an id is an error.
8. **The rules of the project stay rules.** `resume` needs the person's own words (`words` is required); a held id also forbids a `run`; a run that may feed a verdict needs its card; the TEST split needs its logged row. An interface that made them optional would be wrong.

Where the other suites stand:
- **autogt** has the class `AutoGT` and the module functions `generate_bev_gt`, `generate_lane_gt`, `gt_key`; its commands are hyphenated (`generate-bev`), each one call of a function. Open from
  here: all commands as members of the class, `--json` on every command, a `status` of a running generation. The GT signature is a hash of the generator's source, so changes belong with the GT chat.
- **evalsuite** has the class `Evaluation` since 2026-11-10 (the section above): `check` / `predict` / `score` / `run` and the reading methods, hyphenated commands, `--json` with nothing else on
  stdout, exit codes 0 / 1 / 2, `EVALSUITE_TRACEBACK=1`. What it lacks, seen from here: a queue (the trainsuite one does not know evaluation jobs) and a `status` of a running `run`. The older
  adapter commands (`qual`, `quant`, `list`, `probe`, `selftest`) keep their words; a change belongs with the evaluation chat.

## Tools in one line each
```
python tools/run.py --id E240.a --hyp "..." --pred "..." --rule "..." [--split test] [--after E239.a] [--gpu | --gpu-max-mb 9000] [--timeout 180] -- <command>   # snapshot + tag lcr/E240.a, heartbeat, GPU slot
python tools/ledger.py add|note|render ...   |   ledger.py tail [--loop 19] [-n 15]   # last rows + runs in progress   |   ledger.py sync   # rows of finished runs that have none
python tools/metrics.py [--loop 18] [-v] [--md]             # per-loop table (also at the head of LEDGER.md); -v shows how each row was counted
python tools/guard_hook.py grant evaluation/golden.json --decision D8 --hours 12 --note "..."   |   guard_hook.py grants | check <path>
python tools/machine_lock.py take timing --hours 2 --exclusive --idle --note "E253"   |   release timing | show | wait <owner>   # runs of other owners wait; start your own with AR_LOCK_OWNER=timing
python tools/decide.py --experiment E240 [--write]          # the rule clauses of the card of experiments/E240_*/experiment.md on the metric cards -> KEEP / CUT / TIE / INCONCLUSIVE (refused when the arms' cards differ in split / frames, or two or more seeds cannot be paired: a card's `seed` or an arm's `seeds` (identity), the experiment's `seeds` (one common order), or `"pairing": "position"`)
python tools/decide.py --variant 0.746,0.741 --control 0.725,0.722 --id E240.a --write   # typed per-seed scores (old interface)
python tools/experiment.py check E240 | show E240 | check --all   # gate G2 as code for the card block of experiment.md (an old experiment.json is still read; `migrate` moves it; schema: schemas/experiment.schema.json)
python tools/card.py check --all | check <card.json> | show <card.json>   # the evaluator contract (schemas/card.schema.json): every metric card decide.py reads
python tools/schedule.py show|next|set|add [--command "..."]    # item states are read from the run records
python -m trainsuite start|queue|status|health|watch|stop ...   # from autoresearch/, AR_PROJECT=<problem>: start, supervise and stop a training (trainsuite/README.md)
python tools/check_numbers.py <report> <sources ...> [--strict]   # --strict: exit 1 when a number is missing from the sources
python tools/check_structure.py
python tools/install_agents.py [--check]
python tools/clean_temp.py loopNN [--move | --delete]
python tools/archdraw.py render <spec.json> | check <spec.json> | --selftest   # architecture drawing <-> code (see "Examples")
python tools/selftest.py
```
All tools in `tools/` are standard-library Python 3 (any interpreter in the WSL or on Windows); `evalsuite/` needs numpy, scipy and matplotlib (the `gcar` environment).

## Examples: two concrete architectures, one level down

The method is the same for every problem; the product differs. Both products below are drawn the same way, and both aim at an end-to-end network: the path from the sensors to the delivered output should be learned or differentiable. The picture marks what is not: **red / orange boxes are classical steps on that path**, grey ones are frozen (cached) features that no loss reaches. The file map next to each picture names the code of every box and, for every red, orange or grey box, what the step does, what it measurably carries (where the repository documents that), which differentiable building block could stand in for it, and the catch.

**bev_fusion**: camera + radar detections + ego motion -> map layers, object maps, boxes, lines at three times. [File map and the end-to-end table](../bev_fusion/training/architecture.md).

![bev_fusion architecture: camera and radar branches fused, a BEV trunk, a time mover and two heads; red boxes are rule steps](../bev_fusion/training/architecture.svg)

**bev_fusion_lane_center**: camera + radar BEV features -> lane-centre-line graph attached to an SD-map route, planner path. [File map and the end-to-end table](../bev_fusion_lane_center/training/architecture.md).

![lane-centre system architecture: frozen foundation and dense head, the builder with its selections, assemble() in numpy, route head and consumer](../bev_fusion_lane_center/training/architecture.svg)

| system | stages from input to output | learned | fixed, differentiable | frozen | rule steps on the path |
|---|---|---|---|---|---|
| bev_fusion (t20 base, drawn 2026-11-03) | 17 | 11 | 2 | 4 | 0 on the main path (the visibility maps are deterministic and cached); two rule steps on the consumer side: the line-cell threshold and the box selection (local maximum, top-k, threshold); the dense map outputs and the soft boxes are network layers |
| lane centre (`champion`, `fast_recall`, `recall`) | 24 | 8 | 2 | 2 | 12: 4 choices inside the builder (seed NMS, child choice, beam top-M, time recursion) and 8 post-processing steps without a gradient path (alive filter, path union, clean-up, junction anchoring, fork relocation, fork pruning, piece cut, Douglas-Peucker for `*_dp`) |

The numbers are counted by `archdraw.py` from the drawing's arrows, so they change when the drawing changes. In both systems the foundation features are frozen and cached; the lane chain's losses are per head and no gradient reaches a head from the delivered graph (its one end-to-end attempt, the joint builder + route fine-tune, was negative three times). The bev_fusion box velocity is not learned in any main-chain run (a dead layer, `bev_fusion/experiments/t11_vel_cause/experiment.md`).

### Keeping drawing and code together

```
python tools/archdraw.py render architecture.json          # spec -> architecture.svg + architecture.md (the file map)
python tools/archdraw.py check ../bev_fusion/training/architecture.json     # exit 1 on any of the following
```
`check` fails when a box names a file or `file::Symbol` that does not exist, when a code file under the covered folders is in no box (or in the spec's `not_drawn` with a reason), when two boxes share a grid cell, an arrow names a missing box, or the picture or the file map is stale against the spec. So the drawing cannot describe code that is gone, code cannot grow without a box, and nobody edits the picture by hand. The spec of each problem sits next to the code it draws (`training/architecture.json`); its `main_path` (the inputs and the outputs) makes the tool count the stages in between by class. Run the three checks after a change in a drawn folder:

```
for s in architecture.json ../bev_fusion/training/architecture.json ../bev_fusion_lane_center/training/architecture.json; do python tools/archdraw.py check $s; done
```
