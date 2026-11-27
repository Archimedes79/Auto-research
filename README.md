# Auto-research

A research system in which AI agents plan and run experiments on perception problems, and code decides what counts as a result. The method
is problem-independent.

| folder | what it is |
|---|---|
| [`autoresearch/`](autoresearch/README.md) | the system: experiments, runs and verdicts (`research/`), the training and evaluation engines, reports, the method, agent roles, templates |
| [`GtGenerator/`](GtGenerator/README.md) | the nuScenes ground-truth generator: BEV ground truth (map layers, objects, velocity) and the lane graph, versioned by signature |
| [`datasets/nuscenes/`](datasets/nuscenes/README.md) | the nuScenes dataset: description, manifest (raw data by hashes), download / check / link tools |
| [`problems/`](problems/README.md) | one folder per research problem: `lane_center` (Spurmitte), `bev_fusion`, and the reference problem `toy` |
| [`champions/`](champions/README.md) | the facts of each problem's best network and its comparison networks |

Git holds code and text. Data and run outputs are where the machine file `~/.research/machine.json` says (on this PC
`D:\Auto-research\data` and `D:\Auto-research\runs`). Not in git either: `temp/` (scratch) and `wsl/` (the disk image of the WSL
distribution). The system before the restructure of 2026-11-27, with its experiments and records, is the git tag `v1`.

## Start here
- The method and its hard rules: [`autoresearch/METHOD.md`](autoresearch/METHOD.md). The commands:
  [`autoresearch/research/README.md`](autoresearch/research/README.md). The tests: [`autoresearch/README.md`](autoresearch/README.md).
- A new problem, metric, picture, run step, report part, ground-truth layer or agent role:
  [`autoresearch/EXTENDING.md`](autoresearch/EXTENDING.md).
- A problem: its `README.md` first. Rules for agents in this checkout: [`CLAUDE.md`](CLAUDE.md).

## A fresh checkout
`git clone` gives the code and the text records. A machine needs:
- WSL2 Ubuntu (or Linux) with the conda environment `gcar` (torch 2.8), and the data ([`datasets/nuscenes/`](datasets/nuscenes/README.md));
- a git identity in the checkout the research system commits in (`git config user.name` / `user.email`, readable by WSL's git);
- a machine file, from `autoresearch/`:
  `python3 -m research machine init --data <dir> --runs <dir> --python gcar=<interpreter> --gpu-slots 2`;
- the agent roles, from the top: `python3 autoresearch/agents/install.py` (copies them into `.claude/agents/`).

The edit hook (`.claude/settings.json`) calls `wsl.exe`: on a machine without Windows it guards nothing.
