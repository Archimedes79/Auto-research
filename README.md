# Auto-research

A research system in which AI agents plan and run experiments on perception problems, and code decides what counts as a result. The method
is problem-independent; three problems use it.

| folder | what it is |
|---|---|
| [`autoresearch/`](autoresearch/README.md) | the system: the core `research/` (experiments, snapshots and runs, the judge, adoption, reports), the training engine `trainkit/`, the evaluation engine `Evaluation/`, the report builder `reporter/`, the method, agent roles and templates |
| [`GtGenerator/`](GtGenerator/README.md) | the nuScenes ground-truth generator: BEV ground truth (map layers, objects, velocity) and the lane graph, versioned by signature |
| [`datasets/nuscenes/`](datasets/nuscenes/README.md) | the nuScenes dataset family: description, manifest (raw data by hashes), download / check / link tools |
| [`problems/lane_center/`](problems/lane_center/README.md) | Spurmitte: a lane-centre-line graph from camera + radar, end to end differentiable; champion E372 S4 |
| [`problems/bev_fusion/`](problems/bev_fusion/README.md) | BEV Fusion: camera + radar BEV perception (maps, objects, motion) on nuScenes |
| [`problems/toy/`](problems/toy/README.md) | the reference problem: small enough for the CPU, the chain test of the system and the template of a new problem |
| [`champions/`](champions/README.md) | the facts of each problem's best network and its comparison networks |

Not in git: data and run outputs (where the machine file `~/.research/machine.json` says; on this PC `D:\Auto-research\data` and
`D:\Auto-research\runs`), `temp/` (scratch) and `wsl/` (the disk image of the WSL distribution). The system before the restructure of
2026-11-27, with all its experiments and records, is the git tag `v1`.

## Start here
- The method and its hard rules: [`autoresearch/METHOD.md`](autoresearch/METHOD.md); the core and its verbs:
  [`autoresearch/research/README.md`](autoresearch/research/README.md); rules for agents in this checkout: [`CLAUDE.md`](CLAUDE.md).
- A problem: its `README.md`, `PROBLEM.md`, `DESIGN.md`, `HANDOVER.md` and `DECISIONS.md`.
- The tests: `python -m research.selftest` (the whole chain on the toy), `python -m Evaluation.selftest`, `python -m trainkit.selftest`
  (from `autoresearch/`, the interpreter with torch), each problem's `tests/`.

## A fresh checkout
`git clone` gives the code and the text records. A machine needs WSL2 Ubuntu (or Linux) with the conda environment `gcar` (torch 2.8;
`datasets/nuscenes/README.md` names how the data are fetched and checked), and a machine file:
`python3 -m research machine init --data <dir> --runs <dir> --python gcar=<interpreter> --gpu-slots 2` (from `autoresearch/`).
