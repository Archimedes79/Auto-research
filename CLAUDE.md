# Research repository (workspace root; the folder name is free: no file in it depends on it)

Four folders live here: `autoresearch/`, `bev_fusion_lane_center/`, `bev_fusion/` and `datasets/NuScenes/`. One git repository holds the code and text of all four (whitelist in `.gitignore`; of `datasets/NuScenes/` only the README and `tools/`, never data); `wsl/` is not in git.

- `autoresearch/` - the general research system: method (`autoresearch/METHOD.md`), agent roles (`autoresearch/agents/`, installed into `.claude/agents/` by `autoresearch/tools/install_agents.py`), templates, prompts and tools (launcher with a git tag per run, ledger, schedule, decisions, checks, cleanup). Problem-independent.
- `bev_fusion_lane_center/` - the problem it is applied to: lane-centre-line graph from camera + radar, attached to an ADASIS-like SD map (tag prefix `lcr`). Its frame is `bev_fusion_lane_center/PROBLEM.md`; its rules, layout and commands:

@bev_fusion_lane_center/CLAUDE.md

If the line above was not expanded, read `bev_fusion_lane_center/CLAUDE.md` and `bev_fusion_lane_center/HANDOVER.md` first.

- `bev_fusion/` - a second problem: a camera + radar BEV system (plan, `training/`, `autogt/`, `evaluation/`, runs, experiments, literature); managed by `autoresearch/` since D23 (2026-10-31: `bev_fusion/project.json`, tag prefix `bf`; trainings start through `run.py`, which calls `temp/train/run_ar.sh` (it trains from the run's snapshot commit and passes on the exit code; `run_cfg.sh` always exits 0) - the start line is in `bev_fusion/records/README.md`; the tools need `AR_PROJECT=../bev_fusion`, the default problem is the lane one); duplicating code from the lane project is allowed. See `bev_fusion/README.md`.
- `datasets/NuScenes/` - description and download / check tools of the raw nuScenes data shared by all problems (README and tools in git, the data never). The raw data itself is on D: (`D:\Auto-research\NuScenes`); derived data (GT, additions, caches) belongs to the problem's project folder on C:. Existing code reads the WSL views `~/data/nuscenes_full` / `~/data/nuscenes` (symlinks to both places). See `datasets/NuScenes/README.md`.

Other things in this folder: `wsl/` is the Ubuntu disk image of the WSL distribution (registered with WSL: do not move, rename or delete, never in git); the scripts and the move log of the folder migration of 2026-10-24 are in `bev_fusion_lane_center/experiment_execution/history/migration_tools/`.
