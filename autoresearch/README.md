# autoresearch - a research system around a product

General and problem-independent: the method (`METHOD.md`), seven agent roles (`agents/`), five templates (`templates/`), two prompts (`prompts/`) and the tools that make the method hold (`tools/`). A problem plugs in through its own folder. It is kept small on purpose: a new template or agent only when the same artifact has been produced twice by hand (`METHOD.md`, "Keeping the system small").

| folder / file | content |
|---|---|
| `METHOD.md` | the cycle P0 frame -> P1 research -> P2 hypotheses + plan -> P3 loop (variant and block experiments) -> P4 close, gates, disruptions, decisions of the person, code and git, roles |
| `agents/` | `ar-scout` (scan and deep mode), `ar-summarizer`, `ar-hypothesis` (six lenses incl. past critic and system review), `ar-referee`, `ar-planner` (plan, schedule, detail plan), `ar-coder`, `ar-reviewer` (code and numbers) - Claude Code subagent format |
| `templates/` | `problem`, `paper_note`, `group_summary` (with its digest), `experiment` (card -> detail plan -> summary, next to `experiment.json`), `plan` (tree + end of loop) |
| `prompts/` | `research_steps.md` (P1 + P2 in four steps started by hand), `loop.md` (P3 + P4, pasted after `/loop`) |
| `tools/` | `run.py` (launcher: ledger row, commit + tag per run, TEST guard, retries), `ledger.py`, `schedule.py`, `decide.py` (the written rule as code), `check_numbers.py`, `check_structure.py`, `clean_temp.py`, `install_agents.py`, `guard_hook.py` (gate hook: Write / Edit of protected files only with a grant under a decided D<n>), `metrics.py` (per-loop table: runs, GPU hours, verdicts, prediction hits, TEST uses), `project.py` (finds the problem and its `project.json`), `selftest.py` - being reworked under decision D8 (see `ROADMAP.md`) |
| `ROADMAP.md` | the reviews of the system (own, independent, and the loop-18 lead's second review with R1-R10 / P1-P10), weaknesses ranked by cost, target packages, the migration, and what the person decided |
| `config.json` | the default problem folder |

## Plug in a problem
1. Make a folder with `PROBLEM.md` (from `templates/problem.md`) and `project.json` (paths: ledger, schedule, runs, TEST log, code folders, tag prefix, scratch folder, structure rules).
2. Set it as default in `config.json` or export `AR_PROJECT=<folder>` for the tools.
3. Set a git identity in the repository's own config (both the Windows and the WSL git use it): `git config user.name ...`, `git config user.email ...`.
4. `python tools/install_agents.py` copies the agents into the workspace's `.claude/agents/` (`--check` tells when the copies drift).
5. List the protected files in `project.json` ("protected") and put the gate hook into the workspace's `.claude/settings.json` (PreToolUse, matcher `Write|Edit|MultiEdit|NotebookEdit`, command `python3 autoresearch/tools/guard_hook.py` run from the workspace; on Windows through `wsl.exe --cd "$CLAUDE_PROJECT_DIR"`). Claude Code reads that file only for sessions started in the workspace folder.
6. Run the research steps by hand (`prompts/research_steps.md`), then loops (`prompts/loop.md`).

Problems using it: `../bev_fusion_lane_center/` (lane-centre-line graph from camera + radar, assignment to an ADASIS-like SD map; tag prefix `lcr`).

## Tools in one line each
```
python tools/run.py --id E240.a --hyp "..." --pred "..." --rule "..." [--split test] [--after E239.a] [--gpu | --gpu-max-mb 9000] [--timeout 180] -- <command>   # snapshot + tag lcr/E240.a, heartbeat, GPU slot
python tools/ledger.py add|note|render ...   |   python tools/ledger.py tail [--loop 19] [-n 15]   # the last rows, for a wake-up
python tools/metrics.py [--loop 18] [-v] [--md]             # per-loop table (also at the head of LEDGER.md); -v shows how each row was counted
python tools/guard_hook.py grant ../autoresearch/tools/ --decision D8 --hours 12 --note "..."   |   guard_hook.py grants | check <path>
python tools/decide.py --experiment E240 [--write]          # the rule clauses of experiments/E240_*/experiment.json on the metric cards -> KEEP / CUT / TIE / INCONCLUSIVE
python tools/decide.py --variant 0.746,0.741 --control 0.725,0.722 --id E240.a --write   # typed per-seed scores (old interface)
python tools/experiment.py check E240 | show E240 | check --all   # gate G2 as code for experiment.json (schema: schemas/experiment.schema.json)
python tools/schedule.py show|next|set|reconcile|add
python tools/check_numbers.py <report> <sources ...> [--strict]   # --strict: exit 1 when a number is missing from the sources
python tools/check_structure.py
python tools/install_agents.py [--check]
python tools/clean_temp.py loopNN [--move | --delete]
python tools/selftest.py
```
All tools are standard-library Python 3 (any interpreter in the WSL or on Windows).
