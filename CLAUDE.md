# Research repository (workspace root; the folder name is free: no file in it depends on it)

Four folders live here: `autoresearch/`, `bev_fusion_lane_center/`, `bev_fusion/` and `datasets/NuScenes/`. One git repository holds the code and text of the first three (whitelist in `.gitignore`); `datasets/` and `wsl/` are not in git.

- `autoresearch/` - the general research system: method (`autoresearch/METHOD.md`), agent roles (`autoresearch/agents/`, installed into `.claude/agents/` by `autoresearch/tools/install_agents.py`), templates, prompts and tools (launcher with a git tag per run, ledger, schedule, decisions, checks, cleanup). Problem-independent.
- `bev_fusion_lane_center/` - the problem it is applied to: lane-centre-line graph from camera + radar, attached to an ADASIS-like SD map (tag prefix `lcr`). Its frame is `bev_fusion_lane_center/PROBLEM.md`; its rules, layout and commands:

@bev_fusion_lane_center/CLAUDE.md

If the line above was not expanded, read `bev_fusion_lane_center/CLAUDE.md` and `bev_fusion_lane_center/HANDOVER.md` first.

- `bev_fusion/` - a second problem, for now only the discussion of a camera + radar BEV system (plan, literature); later its own GT, evaluation and code (duplicating code from the lane project is allowed). See `bev_fusion/README.md`.
- `datasets/NuScenes/` - the raw nuScenes data shared by all problems (description and download / check tools; the data is in the WSL home; not in git). Raw data and a problem's own GT and additions are kept apart: those belong to the problem folder (see `datasets/NuScenes/README.md`).

Other things in this folder: `wsl/` is the Ubuntu disk image of the WSL distribution (registered with WSL: do not move, rename or delete, never in git); the scripts and the move log of the folder migration of 2026-10-24 are in `bev_fusion_lane_center/experiment_execution/history/migration_tools/`.
