# autoresearch - a research system around a product

General and problem-independent: the method (`METHOD.md`), the agent roles (`agents/`), the templates of every artifact (`templates/`), the loop prompt (`prompts/`) and the tools that make the method hold (`tools/`). A problem plugs in through its own folder.

| folder / file | content |
|---|---|
| `METHOD.md` | the cycle P0 frame -> P1 research -> P2 hypotheses + plan -> P3 loop -> P4 close, gates, disruptions, code and git, roles |
| `agents/` | one file per role in Claude Code subagent format (`ar-scout`, `ar-group-summarizer`, `ar-hypothesis`, `ar-past-critic`, `ar-referee`, `ar-planner`, `ar-detail-planner`, ...) |
| `templates/` | problem, topic brief, paper note, group summary, research digest, hypothesis card, experiment plan, schedule, detail plan, experiment summary, loop summary, handover, closed questions |
| `prompts/` | `research_cycle.md` (P1 + P2, run before a loop), `loop.md` (P3, pasted after `/loop`) |
| `tools/` | `run.py` (launcher: ledger row, commit + tag per run, TEST guard, retries), `ledger.py`, `decide.py` (the written rule as code), `check_numbers.py`, `check_structure.py`, `clean_temp.py`, `schedule.py`, `install_agents.py`, `project.py` (finds the problem and its `project.json`), `selftest.py` |
| `config.json` | the default problem folder |

## Plug in a problem
1. Make a folder with `PROBLEM.md` (from `templates/problem.md`: goal, product chain, metrics, protocol, current best, gaps, problem groups, foundation topics) and `project.json` (paths: ledger, schedule, runs, TEST log, code folders, tag prefix, scratch folder, structure rules).
2. Set it as default in `config.json` or export `AR_PROJECT=<folder>` for the tools.
3. `python tools/install_agents.py` copies the agents into the workspace's `.claude/agents/`.
4. Run a research cycle (`prompts/research_cycle.md`), then loops (`prompts/loop.md`).

Problems using it: `../bev_fusion_lane_center/` (lane-centre-line graph from camera + radar, assignment to an ADASIS-like SD map; tag prefix `lcr`).

## Tools in one line each
```
python tools/run.py --id E240.a --hyp "..." --pred "..." --rule "..." [--split test] [--after E239.a] -- <command>   # commits + tags lcr/E240.a
python tools/ledger.py add|note|render ...
python tools/decide.py --variant 0.746,0.741 --control 0.725,0.722 --id E240.a --write
python tools/schedule.py show|next|set <item> <state>
python tools/check_numbers.py <report> <sources ...>
python tools/check_structure.py
python tools/clean_temp.py loopNN [--move | --delete]
python tools/selftest.py
```
All tools are standard-library Python 3 (any interpreter in the WSL or on Windows).
