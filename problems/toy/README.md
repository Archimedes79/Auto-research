# toy - the reference problem of the research system (P0)

A disc drifts through a noisy 32 x 32 picture; a small network marks its pixels. It is small enough to run on the CPU in minutes and goes through every activity with the general tools: ground truth, training, watching, running, evaluation (numbers, pictures, compute time), a verdict and a report. It is the chain test of the system and the template for a new problem. Plan: `../../autoresearch/P0_PLAN_2026-11-26.md`; moved here from `autoresearch/examples/toy/` on 2026-11-26 (step S0 of `../../autoresearch/INTEGRATION_2026-11-26.md`). It still runs on today's tools; step S1 rebuilds it on the new core and fixes the review findings listed there.

## Run it (from this folder, under WSL)
| activity | command | output |
|---|---|---|
| data and truth | `bash tools/prepare.sh make training/configs/base.json` (`have` instead of `make`: is it there?) | `data/store/raw/`, `data/store/gt/<signature>/` |
| look at the truth | `bash tools/py.sh gt/generate.py view --out runs/gt/truth.png`; known answers: `... generate.py check` | a picture |
| train (a diagnostic) | job json with `"no_card": true`, then from `autoresearch/`: `python3 -m Training run <job> --project ../problems/toy` (preset `smoke`: 40 steps) | `runs/<run>/` |
| watch | `python3 -m Training status <id> --project ../problems/toy` (also `plot`; `python3 -m Training --root ../problems/toy/runs health <run>`) | text, a curve picture |
| an experiment | `python3 experiments/E1_width/launch.py` (`--check`: preview). One run per arm and seed: train, then evaluate | records, tags `toy/E1.*`, `results/cards/<run>/` |
| evaluate by hand | `bash tools/py.sh evaluation/score.py runs/<run>/config.json` | the same, but into `temp/cards/`: a call without a code commit never adds or replaces a record (also true for runs started with `--no-git`) |
| verdict | from `autoresearch/`: `AR_PROJECT=../problems/toy python3 tools/decide.py --experiment E1 [--write]` | `decision.json`, `results.md`, the ledger |
| report | `bash tools/py.sh reports/E1/cost.py`, then `bash tools/py.sh ../../reporter/reporter.py build reports/E1 --strict` and `... check reports/E1 --strict` | `reports/E1/report.html` |
| test everything | `bash tools/py.sh tests/selftest.py` (`--quick`: known answers only) | a throw-away copy runs the whole chain, about 80 s |

`tools/py.sh` runs a script with the interpreter that has torch (`TOY_PY`, default the `gcar` environment) and puts this folder and `autoresearch/` on the path.

## Layout
| folder | holds |
|---|---|
| `data/` | the recipe of the raw data and its manifest (shards with split, size, hash). The splits are in the manifest |
| `gt/` | the truth generator: `raster.py` is the hashed part, the store is named by the signature |
| `net/` | `prep/` the shard reader, `model/` the network, `loss/` the loss |
| `training/` | the trainer on `trainkit`, `configs/`, the drawing of the system |
| `runner/` | the Runner: checkpoint and frames to stored predictions |
| `evaluation/` | the adapter (metric, views, card, timing) and `score.py`, the evaluation step of a run |
| `experiments/` | one folder per experiment: card, summary, its own code |
| `records/`, `results/` | written by the general tools: ledger, run records, metric cards |
| `reports/` | one folder per report |
| `tests/`, `tools/` | the selftest; the three wrappers |
| `runs/`, `temp/` | heavy outputs and scratch, not in git |

## What E1 gave
Verdict KEEP by the card's rule; the narrow control did not finish training, so it says nothing about the number of channels (`experiments/E1_width/experiment.md`). E1 ran before the adapter learned to leave out undefined ratios. Its numbers are all defined and unaffected, but its evaluations carry the older evaluator hash: a new evaluation compares with them only after both are scored again.

## Frictions: where the general tools did not fit (found while building, 2026-11-26)
| # | tool | friction | what the toy does |
|---|---|---|---|
| 1 | `Training/compose.py` | every start gets `--gpu --split train`: a CPU job takes a GPU slot | an own slot store, so no real GPU slot is taken |
| 2 | `Training` prepare | one build run per job, so four arms would build one store four times; items cannot wait for each other | one combined item; the experiment's launcher builds once before the jobs |
| 3 | `Evaluation.run` / `score` | the default metric names are the BEV ones (`maps`, `boxes`) | passes `metrics=("seg",)` |
| 4 | Runner | every problem writes its own; the toy's repeats most of `bev_fusion/runner/runner.py` | `runner/runner.py`, a candidate to move into `Evaluation` at the next user |
| 5 | `reporter` | a value key splits at dots, so a name like `seg.iou` cannot be addressed | the card uses names without dots |
| 6 | card contract | `card.py` refuses NaN; an undefined ratio (nothing predicted) must be left out | the adapter drops it (found by the selftest) |
| 7 | `experiment.py check` | a card that names its result files is NOT READY before the runs, although naming them before the start is what registers the arms | the card names them; the check passes after the runs |
| 8 | kill rule | enforced only while a watcher runs; `decide.py` does not look at it. In E1 it also watched the wrong number | named in E1's summary |
| 9 | `result.py` | the conclusion has two places (`experiment.md` section 3 and `results.md`) | written once, in `experiment.md` |
| 10 | `card.py check --all` | other json files in the cards folder (timing, worst frames) are listed as skipped | left as is |
| 11 | `guard_hook.py`, `commit_paths.py`, `package.py` | a problem below the top level of the repository is invisible to them | the toy has nothing protected |
| 12 | root `.gitignore` | `*.svg` is ignored: a problem's drawing needs an exception | one line in this folder's `.gitignore` |
| 13 | `run.py` | writes `results/<id>/provenance.json` for diagnostic runs too | kept |
| 14 | timing | inside parallel runs the machine is not quiet | the report measures again, one network after the other |
| 15 | `run.py` | the runs execute the working tree, the tag only records it (one tag and commit per run) | as the lane problem today |
| 16 | `trainkit` | no default timing of the wait for data | shown as `-` in the report |
| 17 | wrappers | the interpreter path of this machine is a default (`server.md` row 2) | `TOY_PY` overrides |

## Not built
P0.6 (two ranks and data shards in `trainkit`) and P0.7 (the toy in the system's selftest, no import of a problem, no data-set names in the general packages).
