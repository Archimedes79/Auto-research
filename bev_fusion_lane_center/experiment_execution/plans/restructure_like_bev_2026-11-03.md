# Lane project: the BEV system's way of working, copied and adapted (Mirko, 2026-11-03)

His order: use a system similar to the BEV system's, the same suites, copy what is project-specific from `bev_fusion/` into this folder and adapt it, take out what is not needed, and go on building the head (the grid chain, card E271). The shared toolchain stays in `../autoresearch/` (`run.py`, `trainsuite`, `evalsuite`, `ledger`, `decide`, `keep3`, `package`, `archdraw`, `commit_paths`); nothing of it is copied. Tag of the state before: `lcr/pre-restructure-2026-11-03` (everything removed below stays reachable through it and through git history).

## 1. What the BEV system has, and what this project does about it
| BEV piece (path in `bev_fusion/`) | what it gives | here | step |
|---|---|---|---|
| trainer contract in `training/train.py` (`log.jsonl`, `config.json`, `CONTROL` / `STOPPED` with exit 75, `.done` only for 0, resume) | `python -m trainsuite status / health / curves / stop / watch / queue` work on a run | the general trainer `training/trainer.py` gets the same contract (the 30 lines are copied, not the 1,500-line trainer) | 2 |
| `training/probe.py` (gradient and update size per module, data wait, incident dump) | the module view of `trainsuite health` | copied unchanged (it knows no project code) | 2 |
| `training/configs/*.json` + presets, `training/startable.py` (static check, smoke twin, start lines) | a run is a config file; a start that cannot work is refused before it takes the GPU | `training/configs/` for the head, `startable.py` adapted to the head's config (no BEV model / loss classes) | 3 |
| `training/interface.py` (contract of what the network reads and returns) | no silent tensor | `training/models/p16_interface.py` exists; the head gets its own contract next to it (`training/grid/interface.py`) | 3 |
| `evaluation/bev_gtcheck.py` (the oracle: ground truth through the chain reproduces the ground truth) | the first test of a chain | `evaluation/lane_gtcheck.py` for the grid chain (stages 1 - 4 on GT lanes) | 3 |
| `evaluation/bev_modules.py`, `bev_timing.py` (module probe on real frames, ms per module on an idle GPU) | where the quality and the time go | later, when the head trains: lane versions in `evaluation/` (the idle-GPU guard is copied from `bev_timing.py`) | 5 |
| `evaluation/board.py` + `records/BOARD.md`, `records/champion.json` | one table of all arms against the base | later, with two trained arms; reads `runs/eval/<name>/*.json` | 5 |
| `FROZEN.md`, `frozen.json`, `tests/test_frozen.py` | the frame is fixed, work goes into modules | after the first head baseline (E271), not before: the frame is not stable yet | 6 |
| `tests/run_gates.sh` | fast gates | `tools/maintenance/run_all_tests.sh` stays the one test runner (it already has the metric, builder and tool tests); a fast subset is added | 4 |
| `runner/bev_runner.py` (the five-layer system on frames without GT) | the product on new frames | later; `lane_prototype.py` is the lane equivalent today | - |
| `evaluation/bev_suite.py` + `bev_task.py` + `bev_adapter.py` | the evalsuite adapter | exists: `evaluation/lane_suite.py` + `lane_task/` | done |
| `training/architecture.{json,svg,md}` by `archdraw.py` | the picture from the code | exists (`training/architecture.json`); the head's boxes are added when it has code | 3 |

Not copied, because the BEV system's problem is another one: its model / loss / dataset code, `precompute.py`, `scratch.py` (checkpoint slimming of 1.5 GB runs), `freeze.py` (module freezing; the lane head has no frozen sub-modules yet), the `bev_velocity` / `bev_fusion_area` / `bev_curves` views.

## 2. What is taken out (step 1)
Rule: a python file that no live code, test or document of this project reaches (import graph from the roots: `evaluate.py`, `eval_parts.py`, `lane_suite.py`, the tests, `trainer.py`, `p16_train.py`, `training/grid/`, `p16_*`, `tools/`, `reports/build/`) and that does not explain how a stored number or a cache was made. Removed from the working tree with `git rm`; the tag above and the history keep them.
- `training/legacy/` (the old modules of loops 2 - 8: 25 files, 390 KB; no reference outside comments);
- `evaluation/loops/` scripts that nothing imports (loop 16 ensemble / hysteresis / eval, loop 17 compare / consumer / dense / series, loop 18 fork-prune eval); `evaluation/examples/run_examples.py` (superseded by `lane_suite.py` and `evaluate.py --compare`);
- `evaluation/drivers/p16_fast_val.py`, `p16_pareto.py`, `p16_route_check.py` (superseded by `evaluate.py` and its parts);
- `training/data/p14_chain_data.py` (superseded by `p16_chain_data.py`).
Kept on purpose: `training/data/prep/` (the recipes of the caches, the maintain-the-reproduction-data rule), `oracle_vertices.py` (where the 14.3 comes from), `p18_regis_eval.py` (registration outputs), the trainers of components of the delivered chain (`p10_route_train`, `p14_chain_consumer`, `p17_mapft`, `p18_regis_train`: they are folded into the config-driven trainer when their components are retrained).

## 3. Order and check
1. tag, test run before (done: `temp/loop23/tests_before.txt`), removal, tests after, `check_structure.py`, `archdraw.py check`; documents (README rows, `CLEANUP_LIST.md`) updated in the same commit.
2. trainer contract + `probe.py` (+ a selftest group: stop request, `log.jsonl`, resume).
3. head: config, interface, `lane_gtcheck.py`, `startable.py`; stages 1 + 2 of E271 as a model with a matching-free loss; oracle first, small run, then the card's arms.
4. fast gates; 5. board / module / timing tools with the first two arms; 6. freeze after the baseline.
Each step is one commit of its own paths (`commit_paths.py`), tests green before it. The machine is idle (the BEV trainings are stopped), the lane head may use the GPU through `run.py` (slot rules, D19 lock).

## 4. Status (2026-11-03 10:00)
- Step 1 (removal) done: 39 files, commit 38d7c51; baseline of `run_all_tests.sh` before: all pass except the autoresearch selftest (it sees the live runs of other chats); after: the fast gates, `evaluate.py --golden check --cached-only`, `test_lane_prototype.py` and `p18_regis.py` pass.
- Step 2 (trainer contract + probe) done: 38d7c51; the RNG-on-GPU resume bug found by the experiments chat fixed in 7cd81b8 (selftest group 9 fails without the fix); `python -m trainsuite status` reads the running E271 runs.
- Step 3 (configs, `startable.py`, `run_config.py`, project.json `trainsuite` block) done: 54ebc03; the oracle (G0) is the experiments chat's `training/grid/test_oracle.py`, so no `lane_gtcheck.py` is written.
- Open: steps 4 - 6 (gates are in place; board / module probe / timing / freeze wait for two trained heads), and the alignment with the BEV chat's plan of the same day (trainkit, manifests, split lists): messages sent; lane `trainer.py` stays a subset of the trainkit contract.

## 5. D50 (Mirko, 2026-11-03)
The lane project is a separate codebase from the BEV system (a head on the frozen GaussianCaR features; the BEV system is not its trunk). What was copied from `bev_fusion/` stays as this project's own tooling; the alignment with the BEV chat's plan (trainkit contract, split fields, a shared lane_graph layer) is void, and the holdout-names file was removed again.
