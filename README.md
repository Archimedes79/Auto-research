# Auto-research

A research system in which AI agents plan and run experiments on perception problems, and code decides what counts as a result. The method is problem-independent; two problems are worked on with it.

![autoresearch overview](autoresearch/overview.svg)

One lead agent plans a round and writes the experiment cards. Experiment agents run them, several at a time. Plain files are the memory (plan, ledger, decisions, handover). Code measures and refuses: a run that feeds a verdict goes through the launcher, a verdict comes from a rule written before the run, and the TEST split and the protected files are opened only with a recorded decision of the person.

## Folders
| folder | what it is |
|---|---|
| [`autoresearch/`](autoresearch/README.md) | the research system: method (`METHOD.md`), agent roles, templates, prompts, tools (launcher with a git tag per run, ledger, schedule, decisions, checks), evaluation and training suites |
| [`GtGenerator/`](GtGenerator/README.md) | the nuScenes ground-truth suite: BEV ground truth (map layers, objects, velocity) per keyframe, versioned (`autogt/vN` tags) |
| [`bev_fusion_lane_center/`](bev_fusion_lane_center/README.md) | problem 1: a lane-centre-line graph from camera + radar, attached to an SD map; the system of record is the token champion (`training/architecture.svg`) |
| [`bev_fusion/`](bev_fusion/README.md) | problem 2: a camera + radar BEV perception system on nuScenes: the five-layer framework and the full network (camera + radar + temporal fusion + motion block) on the current GT version; training, evaluation, experiments, literature |
| [`data/NuScenes/`](data/NuScenes/README.md) | the nuScenes dataset family: description, manifest, download / check / link tools and the map of where raw data, caches and checkpoints live (the data itself is not in the repository) |

Not in git: the rest of `data/` (generated data), `temp/` (scratch), `wsl/` (disk image).

## Status (2026-11-21)
- **Method and tools** are in place and tested: the launcher (`run.py`: a ledger row, a done flag, the code commit and a git tag per run), the ledger, the decisions record (one neutral table, `bev_fusion_lane_center/experiment_execution/DECISIONS.md`, read by the gates of both problems), the gates (`guard_hook.py`, `testgate.py`), the four suites `GtGenerator`, `Training`, `Runner` and `Evaluation` (a class with a command line each), the job layer `jobrunner`, and the generated architecture drawings (`archdraw.py`). Each problem keeps a small design spec, `DESIGN.md` (the modules of the chain and the rules it obeys).
- **Problem 1 (lane centre-line graph):** the token champion (D113 - D120): the frozen lane head B' makes tokens, one token head wires and types them end to end; VAL topo F1 0.6682 / 0.6813, 16.2 ms per frame, 2.323 M parameters. The assignment to the SD map is built as a separate stage (E389: the route head reads the map but its gain over the matched no-map head stays below the bar, TIE; E390 builds the map as a logical road graph). Details: [`bev_fusion_lane_center/README.md`](bev_fusion_lane_center/README.md), `PROBLEM.md`, `HANDOVER.md`.
- **Problem 2 (BEV system):** the five-layer framework and the full network are built on GT release v14 (front v13 + lane l6); the experiments t16 - t26 and their outcomes are in the cards and the ledger; the state is in [`bev_fusion/README.md`](bev_fusion/README.md).
- **Ground truth:** [`GtGenerator/README.md`](GtGenerator/README.md) (front v13, lane l6, release v14).
- **Data:** raw nuScenes on D:, caches and checkpoints next to it, reached through links; the map is [`data/NuScenes/README.md`](data/NuScenes/README.md).
- **Clean-up of 2026-11-20 / 21 (D129):** the decisions record was rewritten in neutral wording and merged into one file (`experiment_execution/DECISIONS.md`, shared by the two problems); clutter and dead code were removed (the removed code is only in the git history); the READMEs and drawings were brought to the current state.

## Where to start
- The method and its rules: [`autoresearch/METHOD.md`](autoresearch/METHOD.md) (six hard rules, everything else is a default).
- The picture and the detail drawing: [`autoresearch/overview.md`](autoresearch/overview.md), [`autoresearch/architecture.md`](autoresearch/architecture.md).
- A problem: its README, `project.json` (paths), `DESIGN.md` (the design spec) and, for the lane problem, `HANDOVER.md` (where we are), `PROBLEM.md` (goal) and `experiment_execution/DECISIONS.md` (the decisions of the owner, one neutral table); `bev_fusion/` keeps its plan in `bev_fusion/plan/`.
- Rules for agents working in this checkout: [`CLAUDE.md`](CLAUDE.md).

## Reproducing a result
A result card names the ground-truth version and mode, the evaluation code version, the frames, the raw-data manifest, the model and the environment (`autoresearch/reproducibility.md`, `autoresearch/tools/provenance.py`). Recipes for the raw data stay in git (`data/NuScenes/`).
