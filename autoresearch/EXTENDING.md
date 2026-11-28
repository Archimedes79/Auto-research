# Extending the system

How it grows, one recipe each. The commands run from `autoresearch/` under WSL. The method: `METHOD.md`; what a problem writes and what the
core does: `research/README.md`. The engines' own extension pages are linked, not repeated.

## A new problem
The toy (`../problems/toy/`) is the template. A new problem adds protected files (`problem.json`, `TEST_LOG.md`) and a TEST gate, so it
starts with the owner: a decided row `D<n>` in the root `DECISIONS.md`, then the owner copies the toy and grants the new folder (an agent
does not copy protected files around the edit hook):
```
rsync -a --exclude experiments --exclude ledger.csv --exclude reports ../problems/toy/ ../problems/<name>/     # the owner: code, not records
python3 -m research grant problems/<name>/ --decision D<n> --note "new problem <name>"
```
The folder name is lower-case letters, digits and `_`, starting with a letter. Change:

| file | what |
|---|---|
| `problem.json` | `name` (the folder name), `tag_prefix` (its own); then steps, score, guards, control floor, splits, `protected`, `tests` |
| `training/configs/base.json` | `name`, `data_root` (`{data}/<name>`), then your settings |
| `dataset/`, `gt/` | the data recipe and its manifest (from the problem folder: `PYTHONPATH=.:../../autoresearch ~/miniconda3/envs/gcar/bin/python dataset/make_data.py manifest`), the ground truth; the data-folder fallback in `net/prep/shards.py` and the dataset name in `evaluation/adapter.py` |
| `net/`, `training/`, `evaluation/`, `tests/` | network, loss, trainer, metric, tests (a renamed test also in `problem.json` `tests`) |
| `training/architecture.json` | title and boxes, then `python3 reporter/archdraw.py render ../problems/<name>/training/architecture.json` (the title of `training/architecture.md` is edited by hand) |
| texts | `README.md`, `TEST_LOG.md`; add `PROBLEM.md` (`templates/problem.md`, and name it in `problem.json` `protected`), `DESIGN.md`, `DECISIONS.md`, `HANDOVER.md` |

The first commands (the machine file must name the problem's `python`):
```
python3 -m research tests <name>
python3 -m research commit problems/<name> DECISIONS.md -m "new problem <name>"
python3 -m research new <name> <slug> "<title>"        # experiments/E001_<slug>/experiment.md: fill the card
python3 -m research check <name> E1
python3 -m research start <name> E1
```

## A new metric
- Write it as counts per frame that add up, and numbers reduced from the summed counts: `Evaluation/EXTENSIONS.md`, section 3.
- In the adapter: return it from `task` (only the metrics asked for, so a stored run's anchor still agrees), name it in `METRICS`, add its
  file to the evaluator hash (the files that shape a number).
- For runs: name it in `problem.json` `evaluate.metrics`; a score or guard is `<metric>.<number>`. `problem.json` and, in many problems,
  `evaluation/` are protected: a grant.
- A new file in the problem needs a box (or a `not_drawn` row) in `training/architecture.json`, then `archdraw.py render` (`research tests`
  checks the drawing).
- The evaluator hash changes with any edit of its files (a new view in the adapter too): the judge calls arms scored by different evaluators
  INVALID. Run the control again instead of reusing it (`from`).

## A new picture
- A `View` of the adapter's task, and taps to look inside a network: `Evaluation/EXTENSIONS.md`, sections 4 and 5.
- In a report: `"draw": [{"run", "frame", "view", "as"}]` in `report.json`, drawn from the stored predictions (no network runs).

## A new run step
An entry in `problem.json` `steps`:
```json
"stats": {"run": ["{python}", "training/stats.py", "{config}"], "gpus": 1, "dir": "stats"}
```
- The words may name `{python} {config} {run_dir} {data} {runs} {seed} {split}`. `gpus` above 0 waits for a GPU slot; `gpu_mb` (optional, with
  `gpus`) is the MiB of GPU memory the step needs: it then also waits until that much is free; `dir` (optional) is
  the step's own folder in the run folder. `"run": "evaluate"` is the core's evaluation step.
- It runs when `default_steps` or a card's `steps` name it; a TEST evaluation runs `test_steps`.
- The script runs from the snapshot in the problem folder: exit 0 done, 75 stopped, anything else failed. A long step stops when a `CONTROL`
  file appears in its folder and logs rows `{"step": n, ...}` to `log.jsonl` there, which the card's kill rules read (`trainkit` does both:
  `trainkit/control.py`). It writes into the run folder, never records.
- A setting it reads enters the base config first, with the old behaviour as its default. A trainer on `trainkit` refuses unknown config
  keys: add the field to its config class too (the toy: `TrainConfig` in `training/train.py`).
- A new script needs a box in the problem's drawing (see "A new metric").

## A new report part
- A report is a folder in git (`problems/<name>/reports/<report>/`, or an experiment's `report/`);
  `python3 reporter/reporter.py new <folder>` writes its chapters.
- A part is a markdown file in `parts/`, named in `report.json` `sections` (`id`, `title`, `part`, optionally `svg`); numbers are `{{name}}`
  filled from `values`. The problem's drawing is `"svg": "../../../training/architecture.svg"` (from `reports/<report>/`).
- Build: `python3 -m research report <name> ../problems/<name>/reports/<report>`. Format and rules: `reporter/README.md`.
- A chapter every new report starts with: `CHAPTERS` in `reporter/reporter.py`.

## A new ground-truth layer or dataset
- nuScenes ground truth: [`../GtGenerator/README.md`](../GtGenerator/README.md), "Extending it" (signatures and versions).
- A problem's own ground truth follows the toy (`gt/generate.py`): a store named by a signature of the code that makes it, its options and
  the data manifest; a stored file that differs is refused; the adapter's `Source.describe()` reports `gt_signature`.
- A new dataset: its manifest (hashes) in git, its bytes under the data root, a holdout split in `problem.json` `holdout`; a reader of the
  problem's own calls `Evaluation.defaults.guard(split, <problem folder>)`. The guard is closed in every process that the research core
  did not hand its TEST gate to: the core's `evaluate` step does it itself; a TEST step of the problem's own (`test_steps`), or a worker
  process started by spawn, that reads the holdout through `guard` or `Evaluation` calls
  `Evaluation.defaults.set_holdout_gate(research.gates.test_refusal)` first, else it is refused. A dataset that several problems share
  gets a folder under `../datasets/`.

## A literature topic
The procedure: [`literature/README.md`](literature/README.md). The folder is `../problems/<name>/literature/topics/<slug>/`; nothing to register:
the researcher agents write there, `python3 literature/topic_table.py <name>/<slug>` makes and checks its table.

## A new agent role
- `agents/ar-<name>.md`: a frontmatter (`name`, `description` = when to spawn it, `tools`, optionally `model`) and a short role text without
  machine paths, dates or run ids.
- A row in `agents/README.md` and in the "Roles" table of `METHOD.md`.
- `python3 agents/install.py` lints the roles and copies them into `.claude/agents/` (`--check`: do the copies differ?).
