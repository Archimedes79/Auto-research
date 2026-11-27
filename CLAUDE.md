# Research repository

AI agents plan and run experiments on perception problems; code decides what counts as a result. Answer the owner in their language (German
or English); documents, plans and code comments in English; pronouns they / them unless stated.

| where | what |
|---|---|
| `README.md` | the folders and a fresh checkout |
| `autoresearch/METHOD.md` | the method, the hard rules, the roles |
| `autoresearch/research/README.md` | the core and its verbs (`research <verb>` below) |
| `autoresearch/EXTENDING.md` | a new problem, metric, picture, run step, report part, ground truth or agent role |
| `GtGenerator/AGENTS.md` | read first for ground-truth work |
| `DECISIONS.md` | the owner's decisions that concern more than one problem; a problem's own are in its folder |

Code and text are in git; data and run outputs never: they are where the machine file says (`research machine`). `temp/` (scratch) and
`wsl/` (the WSL disk image: never move or delete) are not in git.

## Working here
- Read first: the problem's `HANDOVER.md`, `PROBLEM.md`, `DESIGN.md`, `DECISIONS.md`; then `research status <problem>`.
- Commands run under WSL (Ubuntu, conda env `gcar`) from `autoresearch/`: `python3 -m research <verb>`; a problem's own scripts with
  `~/miniconda3/envs/gcar/bin/python` from the problem folder, with the module path of its `problem.json`.
- An experiment: `research new` -> the card -> `ar-critic` (cards) -> code default-off, committed, with its test -> `research check` ->
  `research start` -> `research judge --write`; an adoption only on the owner's decision (`research adopt`).
- After a change of a problem's code: `research tests <problem>` and `research anchor <problem> <champion run>` (for `lane_center` a change
  of the network is checked by a new run: the anchor only re-scores stored predictions).
- Commit own paths only (`research commit <paths> -m ...`), never `git add -A` in this shared checkout; push only when asked, after
  `research push-check`.
- Protected files change only under a grant of a decided decision (`research grant`); the hook `autoresearch/research/hook.py` refuses
  otherwise. A refusal is a question for the owner, never a reason to go around it.
- Changes of guarded files are handed to the owner as exact edits (file, old text, new text).

## Pitfalls
- PowerShell 5.1 has no `&&`; `wsl bash -c` re-parses `$` and backticks: run scripts from files. Git Bash has no python: use WSL.
- Never edit a script a running job executes; never kill another chat's job (`research status`, `ps`).
- A conversation rollback deletes files its session created: commit what matters.
