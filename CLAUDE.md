# ML-Experiment (workspace root)

Two things live here, and one git repository holds the code of both (whitelist in `.gitignore`):

- `autoresearch/` - the general research system: method (`autoresearch/METHOD.md`), agent roles (`autoresearch/agents/`, installed into `.claude/agents/` by `autoresearch/tools/install_agents.py`), templates, prompts and tools (launcher with a git tag per run, ledger, schedule, decisions, checks, cleanup). Problem-independent.
- `bev_fusion_lane_center/` - the problem it is applied to: lane-centre-line graph from camera + radar, attached to an ADASIS-like SD map (tag prefix `lcr`). Its frame is `bev_fusion_lane_center/PROBLEM.md`; its rules, layout and commands:

@bev_fusion_lane_center/CLAUDE.md

If the line above was not expanded, read `bev_fusion_lane_center/CLAUDE.md` and `bev_fusion_lane_center/HANDOVER.md` first.

Other things in this folder: `wsl/` is the Ubuntu disk image of the WSL distribution (registered with WSL: do not move, rename or delete, never in git); the scripts and the move log of the folder migration of 2026-10-24 are in `bev_fusion_lane_center/experiment_execution/history/migration_tools/`.
