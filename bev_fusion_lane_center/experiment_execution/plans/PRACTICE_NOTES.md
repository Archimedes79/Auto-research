# Practice notes - automatic experimentation, and what this project takes from it

You are not the first to run an agent (or a script) in a loop of experiments, and the setup works well for exactly the reason that it is simple. Sources: web searches on 2026-10-24 (result summaries, official pages not opened, so details marked "per summary" should be checked before they are quoted). Nothing was downloaded.

## Published setups

| setup | idea | what to take |
|---|---|---|
| Karpathy "autoresearch" (2026, per summaries of [DataCamp](https://www.datacamp.com/tutorial/guide-to-autoresearch), [InsiderLLM](https://insiderllm.com/guides/karpathy-autoresearch-local-gpu-guide/)) | the human writes `program.md` (direction, rules); a coding agent edits one training file, runs a **fixed 5-minute** budget, keeps the change in git if the validation metric improves and resets if not; every attempt (also crashes) is a row in `results.tsv`; a read-only `prepare.py` is the neutral judge | our `PROGRAM.md` is the same role; add: a fixed budget per comparison, a read-only evaluation (`evaluation/`), one results table for *every* attempt (`LEDGER.csv`), keep / discard by a written rule |
| AI Scientist-v2 (Yamada et al., [arXiv 2504.08066](https://arxiv.org/abs/2504.08066)) | agentic **tree search** over experiments steered by an experiment-manager agent; no hand-written templates; one generated workshop paper passed review, external checks found hallucinated results in it | keep the tree (our ladder D1-D8, successive halving) but verify every number against the files (our numbers-reviewer step); run generated code in a sandbox |
| AIDE (Weco AI, [arXiv 2502.13138](https://arxiv.org/abs/2502.13138)) and MLE-bench ([arXiv 2410.07095](https://arxiv.org/abs/2410.07095)) | solution = node in a tree of scripts; operators *draft / debug / improve*; a summariser keeps the prompt short; benchmark of 75 Kaggle ML-engineering tasks | debugging is a first-class operator: crashed runs get a retry path (`run.py`), not a silent flag |
| AIRA-dojo (Meta / UCL, [arXiv 2507.02554](https://arxiv.org/abs/2507.02554)) | a framework where search policies for research agents can be compared | idea only |
| Tracking / pipeline tools: MLflow, Weights & Biases, DVC, Hydra, Optuna, Ray Tune (from memory) | run records with parameters, metrics, artifacts and code version; config files instead of flags; data / model versioning by hash; sweeps with pruning (Hyperband = successive halving) | we have the ledger, hashes (`models/REGISTRY_ALL.md`), successive halving; no config files yet |

## Practices that make such loops trustworthy (and where this project stands)

| practice | state here |
|---|---|
| hypothesis and decision rule written before the run | yes (plan sections, `templates/experiment_card.md`) |
| one table of *all* attempts, failed ones included | `LEDGER.csv` (from now on); loops 16 / 16b in the ladder cards (`runs/e123_ladder/cards`), loop 17 in `LOOP17_STATE.md` |
| fixed evaluation code the agent cannot change | `evaluation/` + selftests that reproduce stored numbers |
| holdout discipline | TEST once; every use in `evaluation/TEST_USAGE_LOG.md` |
| seeds and noise band | 2 seeds, noise band 0.010 (S), seed spread of the route head up to 0.09 |
| fixed budgets for comparison | ladder fidelity in units `u` (1u, 2u, 4u, 8u), run length doubling per rung |
| run record with command, env, code version | `run.json` (command, env name, python); environment packages in `tools/env/environment_gcar.txt`; code version = `scripts/` is not under git: **gap** |
| completion flag only on success | `run.py`; the false `e142.done` of loop 17 is the reason |
| checkpoint + resume for long trainings | convention (400-500 its); `p17_mapft.py` still to be patched (HANDOVER) |
| configs as files | **gap**: arguments live in wrapper scripts; a `configs/` folder with one json per configuration (e.g. "j1 192/96") is the next cheap step |
| code version control | **gap**: no git in this folder. `git init` + one commit per milestone would give the "ratchet" (keep / reset) of autoresearch and protect against the rollback problem; needs the user's OK |
| automatic regeneration of derived docs | `temp/STRUCTURE.md`, `experiments/README.md`, `LEDGER.md`, `models/REGISTRY_ALL.md` are generated |

## Suggested next steps for the infrastructure (small, in this order)
1. `git init` in `bev_fusion_lane_center` with a `.gitignore` for `runs/`, `data/`, `downloads/`, `_to_delete/`, `*.pt`, `*.npy` (asks the user; no remote).
2. `configs/*.json` for the named configurations (deployment j1 192/96, control, champion) read by `p16_pipeline*.py`.
3. A `tools/experiment/sweep.py` that takes a list of arms and runs them through `run.py` with the GPU guard (the ladder queue scripts do this ad hoc).
4. A nightly `make_structure.py` + `ledger.py render` so the generated docs never go stale.
