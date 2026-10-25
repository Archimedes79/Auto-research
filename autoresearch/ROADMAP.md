# Roadmap of the research system (review of 2026-10-25)

Two reviews merged: the author's own and an independent, read-only one (a fresh agent that was given the files, not the author's conclusions). Its key claims were checked against the files (loop-18 ledger, run records, TEST log, launchers). The steps below become system-improvement cards of cycle 19.

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
- **Open:** D1 radar download, D2 GitHub repository, D3 folder rename, D4 fast deployment variant, D5 purge, D6 start of M0 + M1.

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
