# Research repository

AI agents plan and run experiments on perception problems; code decides what counts as a result. The method, its hard rules and the roles:
`autoresearch/METHOD.md`. The core and its verbs: `autoresearch/research/README.md`. Answer the owner in their language (German or English);
documents, plans and code comments in English; pronouns they / them unless stated.

| folder | what |
|---|---|
| `autoresearch/` | the research system: `research/` (experiments, runs, verdicts), `trainkit/`, `Evaluation/`, `reporter/`, agents, templates |
| `GtGenerator/` | the nuScenes ground-truth generator (`GtGenerator/AGENTS.md` first) |
| `datasets/nuscenes/` | the nuScenes dataset: description, manifest, download / check / link tools |
| `problems/<name>/` | one problem each: `lane_center` (Spurmitte), `bev_fusion`, `toy` (the reference and chain test) |
| `champions/<name>/` | the facts of each problem's best network and its comparison networks (protected) |
| `DECISIONS.md` | the owner's decisions that concern more than one problem; a problem's own are in its folder |

Code and text are in git; data and run outputs never: they are where `~/.research/machine.json` says (`research machine`; on this PC
`D:\Auto-research\data` and `D:\Auto-research\runs`). `temp/` (scratch) and `wsl/` (the WSL disk image: never move or delete) are not in git.

## Working here
- Read first: the problem's `HANDOVER.md`, `PROBLEM.md`, `DESIGN.md`, `DECISIONS.md`; then `research status <problem>`.
- Commands run under WSL (Ubuntu, conda env `gcar`) from `autoresearch/`: `python3 -m research <verb>`; a problem's own scripts with
  `~/miniconda3/envs/gcar/bin/python` from the problem folder, its module path as in its `problem.json`.
- An experiment: `research new` -> the card -> `ar-critic` (cards) -> code default-off, committed, with its test -> `research check` ->
  `research start` -> `research judge --write`; an adoption only on the owner's decision (`research adopt`). After a change of a problem's
  code: `research tests <problem>` (its own tests) and `research anchor <problem> <champion run>` (the champion's numbers still come out).
- Commit own paths only (`research commit <paths> -m ...`), never `git add -A` in this shared checkout; push only when asked, after
  `research push-check`.
- Protected files change only under a grant of a decided decision (`research grant`); the hook `autoresearch/research/hook.py` refuses
  otherwise. A refusal is a question for the owner, never a reason to go around it.

## Pitfalls
- PowerShell 5.1 has no `&&`; `wsl bash -c` re-parses `$` and backticks: run scripts from files. Git Bash has no python: use WSL.
- Never edit a script a running job executes; never kill another chat's job (`research status`, `ps`).
- A conversation rollback deletes files its session created: commit what matters.
