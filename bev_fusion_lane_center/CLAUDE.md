# CLAUDE.md - rules for a session in this project

- Fully differentiable lane-centre-line graph from camera + radar BEV (nuScenes), planner-ready, very good AND very fast
- The user (Mirko) writes German or English: answer in their language; documents, plans and code comments in English; pronouns they / them unless stated
- Live results do not belong here (they are state, see below)
- This folder is one problem of the general research system `../autoresearch/` (method: `../autoresearch/METHOD.md`: P1 research -> P2 hypotheses + plan -> P3 loop -> P4 close; agent roles in `../autoresearch/agents/`; templates in `../autoresearch/templates/`). The problem itself: PROBLEM.md (goal, chain, metrics, current best, gaps, problem groups A1-A9, foundation topics F1-F3); machine-readable paths: project.json
- Two systems: the product (the perception system, `training/`) and the improver around it (`../autoresearch/tools/`, `evaluation/`, `literature/`)

## Read first: the state is five files, nothing else
- HANDOVER.md (<= 3 KB): where we are, what runs, the next step, the owner of the code (the lead chat), pointers; rewrite it, do not append
- experiment_execution/DECISIONS.md: the only channel for Mirko's decisions (champion promotion with its one TEST check, changes of PROBLEM.md, changes of the tools or gates, permissions); write an entry and wait for the recorded answer, never act on it before
- experiment_execution/PLAN.md: the experiment tree of the loop with a status per node
- experiment_execution/schedule.csv (`python3 ../autoresearch/tools/schedule.py show | next | reconcile`): what happens next; a disrupted loop resumes from it
- experiment_execution/LEDGER.md (from LEDGER.csv): one row per run with prediction, rule, result, decision, commit; the journal of a loop goes into ledger notes, not into a new state file
- when a loop starts: the loop prompt (from `../autoresearch/prompts/loop.md`: STOP_AT, TEST / mail / shutdown permissions; read each time, do not re-plan)
- for reference only: README.md (folder map), evaluation/PROTOCOL.md (splits, metrics, timing), experiment_execution/closed_questions.md, literature/OVERVIEW.md before literature work (edit literature/papers.csv, then `wsl python3 tools/literature/catalogue.py all`); old state files are in experiment_execution/history/
- what to read in full: single-file records (ledger, decisions, closed questions, plan) are complete files - search them, do not split or roll them; experiment folders: the last two loops in full, older loops through their loop / weekly summaries and PROBLEM.md's current best

## Evaluate
- every number of a model comes from `tools/env/_py14.sh evaluate.py <registry name> [--compare <name>]` (models/registry.json holds checkpoint, config and the standard settings; a new model = a new registry entry, not a new evaluation script); headline metrics: Q at <= 14.3 vertices per 100 m, topo F1, lanes found, R@1 far, route success, ms per frame (S for the history)

## Git: code and text records, a tag per experiment
- one repository, the parent of this folder (whitelist in its .gitignore; remote `origin` = GitHub archimedes79/Auto-research-lmc, see DECISIONS D2): `autoresearch/`; this folder's code (training/, evaluation/, tools/, reports/build/, temp/ scripts, models/registry.json, PROBLEM.md, project.json, CLAUDE.md, README.md); and all TEXT records (decided 2026-10-25): HANDOVER, experiment_execution/ (plan, schedule, ledger, decisions, closed questions, run records, history), experiments/ (cards, summaries, result files), data/splits.json, runs/eval/ (metric cards), runs/e123_ladder/cards/ (model configs), models/*.md, reports/source/ (digests, reviews), the literature summaries (OVERVIEW, CATALOGUE, papers.csv, register, papers/, deep/)
- never: data, checkpoints, caches, binaries (images, videos, pdf, pptx, html), third-party code (the per-repository code extractions literature/code/<repo>.md are extracted on the fly by the process in literature/code/README.md)
- git history is the long-term memory: nothing committed is rolled away; what rolls is only what an agent reads (rule above)
- `run.py` commits the code when a run starts and tags it `lcr/<run id>` (the ledger and run.json record the commit); `clean_temp.py` commits and tags the scratch (`lcr/loopNN-scratch`) before it removes it, so experiment code is never lost
- main = the reviewed core bundle; a change of shared code gets an `ar-code-reviewer` pass; one lead chat per loop owns the code (named in HANDOVER.md), other chats work in literature/ or reports/ or only read (separate worktrees need roadmap step M7 first)

## Layout rules
- keep it simple: for a one-time purpose a throw-away script and temporary files are better than general code. Scratch goes to temp/loopNN/; when the loop is filed run `wsl python3 ../autoresearch/tools/clean_temp.py loopNN` (dry run, then --move to _to_delete/ or --delete; it refuses if live code still imports a module from there). Build a reusable tool only when it is used a second time
- no files in the project root; one-off work in temp/ (nothing in training/ imports it); reusable helpers in tools/ with docstring, selftest and a row in tools/README.md
- metrics live in evaluation/: extend them, never write a private metric; after changes in a code folder run tools/maintenance/run_all_tests.sh
- keep the structure: a move or a new file in a folder means updating that folder's README.md in the same step; at the end of every loop run `wsl python3 ../autoresearch/tools/check_structure.py` (root files, README per folder, dead paths in the living documents; rules in project.json) and fix what it reports; open cleanup items live in tools/maintenance/CLEANUP_LIST.md
- evaluation/: metric libraries (sparse_lanes, lane_grid), drivers/ = general scripts, loops/loopNN/ = scripts of one loop; training/legacy/ = old modules that live code still imports; temp/ holds only the running loop's scratch
- experiment_execution/PLAN.md = the experiment tree of the loop (written in P2 from `../autoresearch/templates/experiment_plan.md`); details of each experiment in experiments/E<id>_<slug>/ (card, detail plan, summary from the templates)
- nothing is deleted: move to _to_delete/ and list it there; never delete raw data or checkpoints

## Names and ids
- loop<NN> in text, p<NN>_ in code; loop 17 ran as E130-E144 and is catalogued as E193-E212
- experiment E<number> (never reused, letters for variants), folder E<3 digits><letter>_<slug>, run E<id>.<arm>, run dir runs/e<id>_<tag>; technique K<n> in the register before use
- run output: log.txt, status.json / metrics.json, model; the .done flag only for exit code 0, a false flag is renamed .done.FALSE
- files: lower-case ASCII snake_case, dates YYYY-MM-DD; every folder humans open has a README.md

## Run
- no absolute paths anywhere: work from this folder (the problem folder); a `wsl` call starts in the matching Linux directory, so every command uses relative paths. Scripts find their root from their own location (shell: `$BFLC` / `$REPO` or `BFLC_ROOT` from tools/env/_py14.sh; Python: `training/bflc_paths.py`; PowerShell: `$PSScriptRoot`), so the repository folder can be renamed or moved. Never type `/mnt/c/...` or `C:\...` into code or documents (records in history/ are left as they were)
- Windows + WSL2 Ubuntu 22.04, conda env gcar, RTX 3060 12 GB: `wsl bash tools/env/_py14.sh <script.py> args` (_pyr.sh retries on CUDA unknown error)
- long jobs: `wsl python3 ../autoresearch/tools/run.py --id ... --hyp ... --pred ... --rule ... --require-card [--after <ids>] -- bash tools/env/_py14.sh <script> ...` in the background; wait with a monitor, do not poll. Never start a run outside run.py (no ledger row, no done flag, no commit and tag); a planned row first: `autoresearch/tools/ledger.py add <id> --hyp ... --pred ... --rule ...`; chains use `--after`, not sleep loops
- the TEST split only as `run.py --split test` after the row in evaluation/TEST_USAGE_LOG.md names the run id (the launcher refuses otherwise)
- at most 2-3 GPU jobs, GPU < 11 GB, checkpoint every 400-500 iterations with resume, timing only on an idle GPU

## Experiments
- card before the run: question, hypothesis, prediction (a number), decision rule (result -> action, noise band), design (arms, controls, seeds, budget), gate, kill criterion
- every experiment: architecture drawing (no red blocks), numbers table with baselines, result images (key ones to the user with SendUserFile), ms per frame, honest negatives
- every finished run gets a ledger row, also crashed and cut ones; catalogue row (E-id); experiment folder
- decide with the written rule, by tool: `autoresearch/tools/decide.py --variant s0,s1 --control s0,s1 [--band 0.010] [--cost-variant ms --cost-control ms] --id <run> --write` gives KEEP / CUT / TIE / INCONCLUSIVE into the ledger; INCONCLUSIVE means add a seed, not adopt; TIE means the cheaper variant
- every wake-up and every milestone: reconcile and update the schedule, the ledger (`ledger.py note <id> --result ... --decision ...`), the PLAN status and HANDOVER.md; a new chat must be able to continue from the files alone
- a question answered the same way twice goes into experiment_execution/closed_questions.md; re-opening it needs a new mechanism
- quality AND speed: ms per frame (idle GPU, batch 1, cached features); an optimisation must reproduce the reference graphs; on a tie inside the noise band take the cheaper variant
- fully differentiable path: no argmax, no matching, no rule step on the training path, gradient reaches every module; better short lane graphs than false ones; say whether a head is pixel- or graph-based
- a loop owns one or two levels of the experiment tree; it ends with updated state files and a handover

## Numbers: seven layers, controls, TEST rules
- layers, cheap first: 0 self-tests of metric and data code; 1 reproduce a stored number; 2 sanity runs (null, oracle, shuffled); 3 DEV / VAL250 proxy; 4 frozen VAL protocol with statistics; 5 TEST once; 6 fresh numbers reviewer. A claim needs the layers its size requires: debugging 0-1, a comparison 3-4, a headline 4-6
- every comparison has controls: null baseline, oracle (a sanity check, not an upper bound), matched reference on the same frames, strong simple baseline, seed twin, equal-budget control
- statistics: at least 2 seeds (route-head spread up to 0.03 on clean frames, 0.05-0.09 under failure), noise band 0.010 in S, paired scene bootstrap (evaluation/sparse_lanes/bootstrap.py); inside the band it is a tie; S rewards sparsity, so report Q (S without the sparsity term) next to it
- numbers come from result files (metrics.json), never typed by hand; every table says VAL or TEST and its denominator; never mix VAL and TEST in one comparison
- TEST (21 scenes): select and tune on VAL only; TEST only once per finished design (winner + control), with the user's OK or a grant in the loop prompt, logged in evaluation/TEST_USAGE_LOG.md before reporting; anything fitted or chosen on TEST is in-sample and must be labelled so; failure, memory and ablation variants are compared on VAL; index test rows with data/splits.json['test']
- before a report is final: `autoresearch/tools/check_numbers.py <report> <card.json / LEDGER.csv ...>` and fix or explain every number it lists; then one fresh `ar-numbers-reviewer` (primary sources, not the author's digest)
- after a change in autoresearch/tools run `wsl python3 ../autoresearch/tools/selftest.py` (it works in a throw-away project; the real ledger is not touched)
- before you believe a number: which split, rows, denominator? tuned or selected on this set? harness self-tested and a stored number reproduced? null baseline? is the oracle really a bound? matched reference? seeds and noise band? interval excludes 0 and exceeds the band? metric gameable? how many comparisons were made? traceable to a run record? reviewed?

## Permissions and safety
- no downloads / installs without asking; never kill another chat's jobs (ListAgents)
- mail and shutdown only if the loop prompt says so (mail: mirko.meuter@aptiv.com via web.de in Chrome, confirm in Sent); check no other chat runs before a shutdown
- rollback hazard: a conversation rollback deletes files its session created with Write / Edit (it destroyed the loop-17 code); cp new code and results when they matter, update HANDOVER.md

## Pitfalls
- GPU: at most 3 jobs, more than 2 extraction streams with workers cause OOM; WSL needs a background sleep 36000 to survive the night
- pkill -f inside wsl bash -c kills its own shell (script file or [p]attern); Start-Process splits quoted args (wrapper .sh); Git Bash rewrites /mnt/... (use PowerShell); no python in Git Bash (use wsl); PowerShell 5.1 has no &&
- heredocs with non-ASCII or \n corrupt files (use Write / Edit); never name a script like a stdlib module; orphan tail -F locks directories
- model.pt saved every N iterations is not a completion sign; a subset-of-perturbations run can overwrite eval.json
- training can derail (BN running_var): lr <= 6e-4, clipping, roll-back guard (trainer.fit); add the builder's time recursion only after the seed heads have learned
- reports: weekly reports and decks are built by reports/build/ (build_week*_report.py, print_weekly.ps1, build_*_pptx.ps1); a PDF open in a viewer cannot be overwritten
