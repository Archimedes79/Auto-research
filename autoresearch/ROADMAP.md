# Roadmap of the research system (review of 2026-10-25)

Two reviews merged: the author's own and an independent, read-only one (a fresh agent that was given the files, not the author's conclusions). Its key claims were checked against the files (loop-18 ledger, run records, TEST log, launchers). The steps below become system-improvement cards of cycle 19.

A third review by the loop-18 lead chat (section "Second review (loop-18 lead)" at the end of this file) adds findings R1-R10 and proposes to build the autonomy core (M6 pulled forward, a queue tick, hooks, `ar metrics`) before the package facades M3-M5; decision D8 in the problem's `DECISIONS.md`.

## Agreed with Mirko (2026-10-25)
The decided entries are recorded in `bev_fusion_lane_center/experiment_execution/DECISIONS.md`.
- **Order:** the safety work (M0-M2) and the cycle-19 research (P1 / P2, no code) run in parallel; loop 19's GPU runs wait for the safety work.
- **Statistics:** 2 seeds for screening; 3 seeds plus the scene-bootstrap interval before a champion proposal.
- **No VAL split now** (data is limited). Instead, doubling the data with the radar parts 4-6 is a separate decision (D1).
- **One lead chat per loop** owns the code.
- **Decisions come to Mirko through `DECISIONS.md`:**
  - promoting a champion, including its one TEST check; TEST is never used otherwise;
  - changes of `PROBLEM.md`;
  - changes of the tools or gates;
  - permissions.
- **The outer loop (P1 / P2) is started by hand, step by step;** the inner loop runs autonomously.
- **Git holds code AND all text records** (no binaries, no data). Single-file records stay complete; agents read the experiment folders of the last two loops in full, older loops through summaries.
- **D8 (a):** the autonomy core comes first, in the order M0 -> P5 -> P1 -> P2 -> P4 -> P3 -> P6-P8; the package facades M3-M5 follow after loop 19 (work of the loop-18 lead chat).
- **Simplification (2026-10-25):** templates 14 -> 5, agents 13 -> 7, one research-steps prompt, and the rule "a new template or agent only when the same artifact was produced twice by hand" (`METHOD.md`, "Keeping the system small"); P6 is done with it.
- **Data:** D1 radar parts 4-6 downloaded; D7 (b) TRAIN2 + CONFIRM + an unseen-location test (feature extraction running, `plans/data_extension_b456.md`).
- **Open** (see `DECISIONS.md`): D3 folder rename, D4 fast deployment variant, D5 purge, D6 start of M0 + M1, D9 launcher git identity (covered by P5).

## Verdict
The design makes sense as a skeleton. Its strengths:
- product and improver are separated, and a problem plugs in through `PROBLEM.md` + `project.json`;
- pre-registration fields exist; the done flag is written only on success;
- one evaluation entry point with a registry, a reproduction check and a paired bootstrap;
- metric-response tests, closed questions, a past-critic lens;
- a code-only git whitelist.

Its weakness is that **almost every rule is prose, and loop 18 shows that prose rules are not followed**:
- 14 of 35 ledger rows have no prediction or rule;
- 0 decisions were made by `decide.py`;
- 12 runs went through opaque shell wrappers;
- one result (E233) has no run record;
- the schedule is empty.

The expensive risks are in records, provenance and decisions, not in the literature pipeline. The order of work is therefore: make records and provenance safe -> make the gates code -> only then build the lean package interfaces.

## Weaknesses, ranked by cost

| # | weakness | evidence | failure scenario |
|---|---|---|---|
| W1 | **records can be lost or corrupted** | `run.py` / `ledger.py` / `schedule.py` rewrite whole CSV files without lock or atomic replace; loop 18 had 36 overlapping run pairs and 2 that ended within one second | one ledger row vanishes; a WSL shutdown mid-write truncates the file (it happened in loop 18) |
| W2 | **no backup of anything but code, and the code is not pushed** | `.gitignore` excludes ledger, run records, TEST log, experiments, `data/splits.json`, the model configs `runs/e123_ladder/cards/*.json` that `evaluate.py` needs; 157 GB cache and 28 GB runs on one disk; `origin` set, `ls-remote` says "not found" | one disk failure loses the research memory and the models |
| W3 | **run-id reuse corrupts the earlier run** | `run.py` deletes the old done flag, overwrites `run.json`, merges into the old ledger row, keeps the old tag; no tool allocates E ids | two chats pick the same id, or a fixed rerun: the record points at code that did not run |
| W4 | **the tag is not the code that ran** | `git add -A` of the whole shared worktree onto the current branch; the job imports the live tree; the fingerprint covers `.py` / `.sh` only; no torch / CUDA / env record (`conda_env: ""` in `run.json`) | another chat's half-done edit is committed and imported (loop-18 TEST job E230.j1p failed on a concurrent edit of `p16_loss.py`) |
| W5 | **decisions are weak and inconsistent** | `decide.py` takes typed numbers, ignores the card, uses one band (0.010, calibrated for S) for every metric; `bootstrap.verdict` (interval beyond the band, only for S) and `decide.py` (seed means) can disagree, no rule says which wins | with 2 seeds and a seed spread near the band, a sizeable share of null comparisons is called KEEP or CUT; the favourable verdict gets picked |
| W6 | **holdouts are reused** | VAL (22 scenes) has carried selections for 18 loops; the loop-18 TEST grant "once" covered several designs and route heads; the TEST log is written by the agent and not protected | over-fitting to VAL; TEST loses its meaning |
| W7 | **gates are prose** | `--require-card` is opt-in; cards are free markdown; the ledger has no card column; the TEST guard of `run.py` checks its own flag while the command picks the split; `evaluate.py` checks the TEST id as a substring | experiments without a rule; TEST used through a wrapper |
| W8 | **stale evaluation cache** | `evaluate.py` cache key = split, run, M, K0 (the loop-18 fallback ignores K0); no checkpoint hash, no inference-code fingerprint; the golden test reads the same cache; `runs/eval/<name>/<split>.json` is overwritten every call | a changed network or builder produces correct-looking old numbers, and the test cannot see it |
| W9 | **no liveness, no budgets** | no PID / heartbeat / timeout in `run.json`; a killed run stays "running" and its `--after` dependents wait 12 h; the GPU guard is check-then-act and treats an nvidia-smi failure as free; the 2-job limit is enforced nowhere; the CUDA retry matches any earlier "CUDA error" in the log | lost GPU hours overnight; three jobs stall each other |
| W10 | **the product is a script collection** | 197 python files, 70 entry points, 128 `sys.path` patches, 19 `--set` scripts, 1 config file; `bflc_env.sh` puts 43 folders on the path and `_py14.sh` picks the first match of `find | head -1` (`selftest.py` exists 3 times) | a scratch copy shadows a live module; every experiment writes a script |
| W11 | **evaluation covers graphs only** | route success, failure series, map error, timing still in per-loop drivers | robustness and route numbers are not one command away |
| W12 | **ceremony and token cost** | 13 roles, 6 lenses, 4 literature levels every cycle, no token / time budget; wake-ups re-read a growing ledger; the planner's gain x probability / cost is never checked against outcomes; the referee is the same model as the authors | paperwork outgrows the experiments |
| W13 | **web-read content is trusted** | `ar-scout` / `ar-deep-reader` have WebFetch + unrestricted Write; nothing says fetched text is data; their notes feed agents with Bash | a page steers a plan |
| W14 | **worktrees per chat cannot work yet** | the state (runs, data, ledger) lives in the checkout; `project.py` and `bflc_paths.py` resolve the root from the checkout | a worktree has an empty ledger and no checkpoints |

Smaller:
- `check_numbers.py` checks decimals only, matches anything in a large pool, ignores sign flips and always exits 0.
- `clean_temp.py --delete` removes the folder for good, although only its `.py` / `.sh` files are in the tag.
- Exit codes are overloaded.
- `install_agents.py` copies the agent files, so the two copies can drift.
- `check_structure.py` skips the experiment folders (E233 links a moved file).

## What is missing
- **Gates as code**:
  - a machine-readable `experiment.json` per experiment; `run.py` refuses without it by default;
  - `decide` reads the metric cards and the rule from that file;
  - a card column in the ledger.
- **Statistics as code**: one decision rule for each metric (band per metric from the measured seed spread). It requires seed agreement AND a scene-bootstrap interval beyond the band, and it counts the comparisons made per loop.
- **Holdout hygiene**:
  - TEST grants in a file only the person writes (a deny rule for agents on the TEST log and grants);
  - a grant is consumed on use;
  - TEST ground truth is read only through the guarded evaluator;
  - a VAL-confirm subset never used for tuning.
- **Human gates** for: promoting a champion (a `registry.json` change), changing `autoresearch/tools` or the gates, changing `PROBLEM.md`.
- **Method feedback** (`ar metrics`):
  - hit rate per lens;
  - predicted vs measured effect;
  - share of KEEPs that survive replication or TEST;
  - GPU hours and tokens per decision.
- **Budgets**: tokens per phase, GPU hours and the maximum number of comparisons per loop, a timeout per run, kill criteria as code; a light cycle by default (P1 only where the gap or the literature changed, three lenses, a full cycle every third time); about 20 % exploration budget.
- **Backups**: push the code; copy the records nightly to a second place; keep one copy of the feature cache and of the registered checkpoints; test a restore.
- **Safety of web content**: scouts return text and write only into `literature/` (a hook or no Write); fetched text is data; the referee spot-checks "checked" facts.
- **Environment record**: torch / CUDA / driver and a hash of the conda environment in every run record. Also an **id allocator** and a **single writer** for each shared state file.

## Target: two packages, one state root
Specs are JSON: the tools stay standard-library on the WSL Python 3.10 (no tomllib, no PyYAML).

**The run folder is the unit of truth.** The ledger and schedule views are rebuilt from the run folders. The few shared files get a lock and atomic replace.

### `ar` (`autoresearch/ar/`, problem-independent)
```
ar new E --title "..."            allocate the next E id atomically, create experiments/E<id>_<slug>/ with experiment.json + card
ar check card E240                gate G2 as code (prediction, rule, metric, band, controls, budget, kill)
ar run E240 [--arm a --seed 0]    expand arms x seeds into runs E240.a.s0 ...; each: snapshot, tag, run from the snapshot, heartbeat, timeout, metric card
ar run --id X -- <command>        the plain launcher (today's run.py)
ar status                         running runs with heartbeat age, ready items, dirty tree, grants left, comparisons this loop
ar decide E240                    seed agreement AND bootstrap interval beyond the metric's band -> decision.json + ledger view
ar check all                      structure (incl. experiment folders), cards, numbers of the last report, golden numbers, record consistency
ar clean loop19 | ar backup | ar metrics
```
Python API: `Project.load()`, `run(exp, arm, seed) -> RunDir`, `decide(exp_id) -> Verdict`.

Experiment file (`experiments/E240_x/experiment.json`):
```json
{"id": "E240", "card": "H19.3", "metric": "Q", "direction": "higher", "band": 0.010,
 "prediction": {"value": 0.745, "vs": "c"},
 "arms": {"a": {"recipe": "builder_ft", "base": "champion", "set": {"w.rec": 3}},
          "c": {"recipe": "builder_ft", "base": "champion", "set": {"w.rec": 1}}},
 "seeds": [0, 1], "budget_gpu_h": 2, "kill": "Q of arm a below c - 0.02 at 50 % of the budget",
 "evaluate": {"split": "val", "compare": "c"}}
```
Snapshot without touching the shared index or moving a branch:
1. `GIT_INDEX_FILE=<tmp> git add -A <code pathspec from project.json>`;
2. `git write-tree`, then `git commit-tree`;
3. `git tag lcr/<id>`;
4. `git archive` of that commit into `runs/<id>/code/`, and the job runs from there.

### `lcr` (`bev_fusion_lane_center/lcr/`, a facade; the `p16_*` modules stay as they are)
```python
lcr.paths                 code root from the file; state root (data, runs, records) from an environment variable -> worktrees work
lcr.registry.get(name)    models/registry.json
lcr.evaluate(model, split="val", compare=None, conditions=None, map=None, metrics=("graph",)) -> card
                          card with a schema version; cache key = sha(checkpoint) + inference-code fingerprint + settings;
                          per-frame records kept so any later paired comparison needs no recompute
lcr.train(recipe, base, overrides, seed, budget, out) -> run_dir     a recipe table over the 2-3 trainers in use (on trainer.fit)
python -m lcr eval | train | time | golden
```

## Migration (each step: `run_all_tests.sh` and the golden numbers pass unchanged)

| step | what | done when | effort |
|---|---|---|---|
| M0 | **golden numbers**: a golden file for all registry names on VAL (CPU, cached) plus one uncached champion re-inference on VAL250, so the network path is covered too | `lcr golden` / `evaluate.py --expect` checks all of them | 2 h |
| M1 | **hygiene in the current scripts**: lock + atomic writes; `--require-card` on by default; refuse id reuse unless the code hash matches; snapshot through a temporary index with a pathspec; PID, heartbeat, timeout; a GPU slot lock (max 2); a precise CUDA retry; environment record; cache key with checkpoint and code hash (rebuild the cache and get the same champion numbers); TEST id matched as a whole id in `evaluate.py` | selftest covers concurrent writes, id reuse, a killed run, a stale cache | 1 day |
| M2 | **backups and push** (needs the person's OK and a reachable remote): push code + tags; nightly copy of the records; one copy of the registered checkpoints and the splits / configs | restore test | 2 h + decisions |
| M3 | tools become the `ar` package; old paths stay as 3-line shims; same selftest | selftest unchanged | 0.5 day |
| M4 | `lcr.paths`, `lcr.registry`, `lcr.evaluate`; `evaluation/evaluate.py` becomes a shim; then conditions (failure series, map error, route success, timing) behind the same call | golden numbers bit-identical; E213 / E220 / E216 numbers reproduced from cards | 1.5 days |
| M5 | `lcr.train` recipes that call the trainers in use (subprocesses first), on `trainer.fit` with resume | a 1 u fine-tune reproduces E219.j1eq within the noise band | 1 day |
| M6 | `experiment.json` with `ar new / check card / run / decide`; first used beside the old path on cycle 19's first experiment, decisions compared | same decision both ways | 1 day |
| M7 | run from the snapshot + state root; cut the PYTHONPATH to `training/` and the evaluation libraries; protected TEST grants; `ar metrics`; agent write limits | a worktree run works; a scratch module can no longer shadow a live one | 1 day |

Do not rename the `p16_*` modules and do not rewrite the trainers wholesale. New experiments use `lcr` and `ar` from M5 / M6 on; old scripts stay only for reproducing old numbers.

## Method changes (no code)
- Light cycle by default; full cycle every third loop.
- Lenses get slots by their record (`ar metrics`).
- Human gates as listed above.
- One decision rule per metric.
- A comparisons budget per loop.
- STOP_AT stays a work window, but filler comparisons go to a VAL-confirm subset with their own band.

## Second review (loop-18 lead), 2026-10-25

Written by the chat that ran loop 18, as `REVIEW_2026-10-25_loop18_lead.md`; merged here verbatim on 2026-10-25 (decision: Mirko, coordinated with that chat; the file itself is in git history). Ids R1-R10 and P1-P10 are unchanged; `DECISIONS.md` D8 refers to them. Two paths it names were renamed by the simplification of the same day and point to their successors; P6 (two experiment classes, light cycle by default) is done in `METHOD.md` and `prompts/` with that simplification.

A second, independent look at the restructuring (`METHOD.md`, `ROADMAP.md`, `tools/`, `agents/`, the problem folder's five state files), written by the chat that ran loop 18 inside the old layout and watched the new one being built around it. The question is the one Mirko asked: does this move us towards **a system that improves code or ML systems by itself, inside a systematic framework**? `ROADMAP.md` already lists fourteen weaknesses (W1-W14) and a migration M0-M7; this review does not repeat them. It adds what the roadmap does not say, disagrees where it should, and ends with ranked proposals.

Sources: the files named above, `experiment_execution/LEDGER.csv` (35 rows of loop 18), the run records, `experiments/E2xx`, `plans/System_Review_2026-10-25.md`, `plans/PRACTICE_NOTES.md`. Checks run today: `check_structure.py` reports "in step"; `selftest.py` passes.

### 1. Verdict

The restructuring is the right skeleton: product and improver are separated, state is five files, decisions have one channel, every run gets a record and (from now on) a tag, one evaluator with a registry, closed questions, a TEST guard. Loop 18 was the last loop without it and paid for that (a truncated ledger, a TEST job broken by a concurrent edit, three jobs stalling each other, an early shutdown). Three of the roadmap's expensive risks (W1, W3, W4) I met personally; its order "records and provenance first" is right.

What the skeleton still lacks, measured against the goal, is the **inner loop**: the method describes a document pipeline (13 roles, four literature levels, cards, plans, detail plans, schedules, summaries) around an experiment step that is still done by hand. In loop 18, 35 runs went through 12 hand-written shell wrappers, the schedule stayed empty, and no decision was made by `decide.py`. A system improves something by itself when the step "variant -> run -> card -> verdict -> next variant" needs no human and no hand-written script. That step should be built first; the literature and planning machinery can stay as it is for a while.

### 2. Findings the roadmap does not name

| # | finding | evidence | consequence |
|---|---|---|---|
| R1 | **Two kinds of experiment, one procedure.** 19 of the 35 loop-18 rows were trainings; 11 of them were `p16_train.py --set ...` variants (a config, no code). The method sends every card through detail planner -> coder -> reviewer -> launcher -> summary | `LEDGER.csv`; `PLAN.md` nodes R2.6-R2.8 and TEST marked "added" during the loop; `schedule.csv` has only its header | the procedure fits block experiments (new code) and is too heavy for variant runs, so the lead bypasses it; **variant runs should run from a file, unattended** |
| R2 | **Nothing measures the improver.** 21 of 35 rows carry a prediction and a rule (all 21 predictions have a number), yet no tool scores predictions against results, counts KEEPs that survived replication, or GPU hours per decision | ledger columns `prediction`, `result`, `decision`; `ar metrics` is the last item of the roadmap (M7) | the framework cannot say whether a change of the framework helped; the improver does not obey the rule it imposes on the product (prediction before, measurement after) |
| R3 | **Provenance was reset, not repaired.** The repository has 7 commits and one tag (`lcr/loop18-end`); none of the 35 loop-18 runs has a tag; the ledger has no `commit` column (its `code_hash` is a fingerprint); 15 experiment READMEs still link `experiment_execution/indexes/LOOP18_STATE.md`, which moved to `history/`; `check_structure.py` says "in step" because it skips `experiments/` | `git tag`, `LEDGER.csv` header, `grep -l indexes/LOOP18_STATE experiments/*/README.md` | "every result traces to the exact code" holds from loop 19 on; say so in `PROBLEM.md` and `experiments/README.md`; extend the structure check to the experiment folders (`history/PATH_MAP.md` has the moves) |
| R4 | **The rule language is known and small.** Every loop-18 rule is a conjunction of threshold clauses: metric, reference arm, threshold, optional condition subset and quantifier ("a - c >= +0.03 in at least two of three failure types AND graph S failure half not lower by more than 0.005") | `rule` column; `decide.py` accepts two seed lists and one band | the roadmap's `experiment.json` is right; its rule field can be a list of clauses, and `decide` can read the metric cards instead of typed numbers, which removes the last hand-typed numbers from the decision |
| R5 | **The scheduler is a document, not a process.** Chaining was done with `--after` and `until [ -f ... ]; do sleep 20; done` inside wrappers; waiting was counted as run time (E232.eval: 5,336 s for an evaluation of minutes) | `temp/launchers/loop18/_run_j1rc6.sh`, `_eval_j3.sh`; `schedule.csv` | a disrupted loop cannot resume from the schedule because nothing was ever in it; a queue needs a tick that starts what fits, not an agent that edits a CSV |
| R6 | **The lead chat is the bottleneck and the single point of failure.** The lead wrote every launcher, every ledger note, every table script; subagents were used for three numbers reviews only (the `ar-*` roles were installed after the loop); two sessions carried the same title, so a message to the peer went to a guess | loop-18 transcript, `temp/analysis/loop18/` | autonomy needs the lead's job reduced to: write the experiment file, run the tick, read the verdicts, write the summary |
| R7 | **Gates can be hooks today.** `.claude/` holds the agents and nothing else; the TEST log, `models/registry.json` and `evaluation/` are writable by any agent; `--require-card` is opt-in | `ls .claude/`; `run.py` | a PreToolUse hook that refuses Write / Edit on a short protected list is an hour of work and makes the three human gates real now, not at M7 |
| R8 | **The decisions channel needs a recorded chat answer.** The rule "act only after the answer is recorded in `DECISIONS.md`" stalls when Mirko answers in chat, which is how every answer so far arrived | `DECISIONS.md` (ten rows decided today, all answered in conversation) | let the agent record the chat answer with time and the quoted words; the file stays the record, the chat the channel |
| R9 | **Memory notes hold results.** My own memory file carries loop-18 numbers; the method says state is five files | `~/.claude/projects/.../memory/bev-lane-loop18-results.md` | rule for agents: memory holds workflow pitfalls and preferences, never numbers; numbers live in the records |
| R10 | **The method is ML-shaped where it could be generic.** Seeds, GPU slots, VAL / TEST and the TEST guard are named as such in `run.py`, `decide.py`, the templates | code and templates | a code system (a library, a compiler flag, a service) has repeats, resource slots, a benchmark and a hidden test set: the same loop with other names; `project.json` already abstracts the paths, the **evaluator contract** (card schema) and the **experiment schema** are what is missing |

### 3. Where I disagree with the roadmap

- **Order.** M3-M5 (the `ar` package rename, the `lcr` facade, `lcr.train` recipes) are refactors: they make the code nicer, they do not make a single run happen without the lead. The autonomy core is M6 (`experiment.json`, `ar run`, `ar decide`) plus a queue tick; it should come right after M0 and the M1 hygiene, before the facades. Loop 19's first variant runs can then already go through it, next to the old path, as M6 intends.
- **`ar metrics` last.** It is the cheapest item (it reads the ledger) and the only one that makes every other change measurable. It belongs before M3.
- **Light cycle "by default".** The roadmap says it under method changes; the prompts still describe the full cycle. Make the full cycle the exception (every third loop, or when the gap under attack changes) in `prompts/research_steps.md` (formerly `research_cycle.md`) itself.
- **Worktree per chat (M7, W14).** With one lead chat per loop and hooks on the protected files, worktrees buy little for the next loops; the snapshot (temporary index + `git archive`, pulled into M1) protects a running job from edits, which was the actual damage.

### 4. Proposals, ranked by autonomy gained per cost

| id | proposal | done when | cost | removes |
|---|---|---|---|---|
| P1 | **`experiment.json` + rule clauses + `ar decide` from cards** (M6 pulled forward). Schema in `autoresearch/schemas/`; arms x seeds; a rule = list of clauses {metric, vs, min_effect or max_loss, band, subset, quantifier}; `decide` reads `runs/eval/<name>/<split>.json` | five loop-18 experiments re-encoded (E215, E219 j1rc / j1eq, E231.a, E232) give the recorded verdicts | 1 day | hand-typed numbers in decisions; free-text rules; R4 |
| P2 | **`ar tick`**: reconcile run records -> start the ready items within the resource slots (max 2 GPU) -> write the "what runs" block of HANDOVER; items name an experiment arm or a plain command; the wake-up is `ar tick` plus judgment | a chain of three items runs unattended from `schedule.csv` with no wrapper; durations exclude waiting; a killed run is marked stalled by heartbeat age | 1 day | shell wrappers (W5 / F5), empty schedule, lost overnight hours (W9); R5 |
| P3 | **`ar metrics`** on the ledger + run records: runs, GPU hours, decided, KEEP, KEEP replicated, prediction hit rate and mean error, comparisons made, tokens where known; loop 18 scored as the baseline | one table per loop in `LEDGER.md`'s head; loop-18 row filled | 0.5 day | R2; the planner's gain x probability never checked (W12) |
| P4 | **Hooks**: PreToolUse guard (`autoresearch/tools/guard_hook.py`, standard library) refusing Write / Edit on `evaluation/TEST_USAGE_LOG.md`, `models/registry.json`, `autoresearch/tools/`, `PROBLEM.md` unless the environment names a decided `D<n>`; `--require-card` default on; scouts write under `literature/` only | a test subagent's edit of the TEST log is refused; `selftest.py` covers the guard | 0.5 day | R7; W6 / W7 / W13 in their cheapest form |
| P5 | **M1 subset met in loop 18**: lock + atomic replace for ledger and schedule; snapshot through a temporary index with the code pathspec, job runs from `runs/<id>/code/`; PID + heartbeat + timeout in `run.json`; refuse id reuse unless the code hash matches; precise CUDA retry | selftest: concurrent writes, killed run, id reuse, concurrent edit during a run | 1 day | W1, W3, W4, W9 |
| P6 | **Two experiment classes in the method**: *variant* (config only: experiment file -> tick -> decide -> one summary line, no detail planner or coder) and *block* (new code: coder + reviewer + spec test before `ar run`); light cycle default in the prompts | `METHOD.md` P3, `prompts/`, `templates/plan.md` (formerly `experiment_plan.md`) say it; loop 19 runs its variants this way | 0.5 day (text) | R1, R6, W12 |
| P7 | **Structure check over `experiments/`**, fix the 15 links with `history/PATH_MAP.md`; provenance note "tags from loop 19" in `PROBLEM.md` / `experiments/README.md`; ledger gets its `commit` column on the next render | `check_structure.py` finds 0 problems with experiments included | 0.5 day | R3 |
| P8 | **Evaluator contract**: `autoresearch/schemas/card.json` (metrics, split, n, denominator, commit, settings, schema version) with a validator; `run.py` and `decide` speak of repeats and resource slots, the TEST guard becomes a holdout guard with the split names from `project.json` | the lcr cards validate; the selftest's throw-away project is a non-ML problem (a benchmark number from a script) and runs the same loop | 0.5 day | R10 |
| P9 | **CONFIRM split and the "adopted" state** (with D7 b): a KEEP on VAL becomes *adopted* only when CONFIRM (39 scenes, never tuned on) agrees inside its band; `ar metrics` counts comparisons per split | rule in `PROTOCOL.md` and `decide`; first used on D4 | text now, code with P1 | W6 |
| P10 | **Early kill from learning curves**: `trainer.fit` writes the VAL250 metric per checkpoint; the tick cuts a run below the control's curve by more than the band at 25-50 % of the budget | replay on loop-16 / 18 curves: fewer than 1 in 10 wrong cuts | 1 day, after P2 and P5 | about 30 % of search GPU hours (System_Review section 4) |

Deferred, not dropped: M3 (package rename with shims), M4 / M5 (`lcr` facades, train recipes). They come after loop 19 has used P1 / P2 once, so the facades wrap what the loop actually calls.

Suggested order: M0 golden numbers -> P5 -> P1 -> P2 -> P4 -> P3 -> P6 / P7 / P8 (text and small code) -> loop 19's first variant experiment both ways (old path and new; same verdict) -> P9 when the new scenes are extracted -> P10. About six working days of tool work before the facades, in parallel with the cycle-19 research as agreed.

### 5. Small items (minutes each)

- `tools/env/_py14.sh` finds a script by name with `find ... | head -1` over five folders including `temp/`: print the resolved path into the run log and refuse when more than one file matches.
- `install_agents.py` copies the agent files; add `--check` (diff against `.claude/agents/`) to `run_all_tests.sh` so the copies cannot drift unnoticed.
- `ledger.py tail --loop 19` for the wake-up instead of the whole `LEDGER.md` (17 KB today; the wake-up set is 65 KB with CLAUDE, METHOD, PROBLEM, PLAN, DECISIONS).
- The loop prompt says "every wake-up (20-25 min)"; with the tick and done flags the wake-up should be event-driven (a monitor on the run records) with a long fallback.
- One title per session role (lead, literature, reports) so cross-session messages are unambiguous; `HANDOVER.md` already names the owner.
- `check_numbers.py` exits 0 always; with P1 the tables of a report come from cards and the check can be a gate for them (System_Review F9).

### 6. What this review did not check

The product code itself (W10, W11 are taken from the roadmap), the literature pipeline's output quality, token costs per phase (no record exists; P3 would start one), and whether `evaluate.py` reproduces the loop-18 TEST cards (M0 is that check).
