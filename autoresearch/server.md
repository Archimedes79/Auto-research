# Moving to a server: what is tied to this machine

The system was built on one Windows box with WSL and one 12 GB GPU. This page says what a move needs, in the order to do it. Nothing here is built ahead of need (`METHOD.md`, "Code, git, size"): each step is done when the first machine of that kind exists.

## What already holds (checked 2026-11-10)
A fresh clone of commit aaac6880 on a plain Linux file system (`/tmp`, another folder name, no `/mnt/c`) passes every selftest: tools (152 s; about 5 minutes on `/mnt/c`), Training (39 groups), jobrunner (183 checks), trainkit, Evaluation (53 checks), the drawing check, and the gate hook refuses a protected file there. The script is `temp/server_check/run.sh` (scratch). Not checked: a real training, a second GPU, a second host, a scheduler.
- **No absolute path, no folder name**: every tool finds the repository from its own file, the problem from `AR_PROJECT` or `config.json`.
- **Plain files are the only state**: the run record, the ledger, the cards. A run's state is read from its heartbeat file, not from a process table, so another host on the same file system reads it right. The locks (`project.locked`) carry the host name and never judge a holder of another host dead.
- **The machine layer is one switch**: `run.py --gpu` (GPU slots, the first-come line, the RAM and memory gates, the settle time). Without `--gpu` the launcher only does what the method needs: card, TEST gate, snapshot and tag, run record, ledger row. That is the form to use under a scheduler.
- **The suites are classes with plain data in and out** (`README.md`, "The interface of the suites"); `jobrunner` turns one call into one job file. A job file is what a scheduler would submit.

## What is tied to this machine
| # | what | where | on a server |
|---|---|---|---|
| 1 | the gate hook is started through `wsl.exe`; on Linux that command fails and its `\|\| true` lets every edit through without a word | `.claude/settings.json` (protected) | the command without the `wsl.exe --cd ...` prefix; a change of a protected file, so the person decides (it is also the open rest of D75) |
| 2 | the interpreter of the numpy / torch parts is `~/miniconda3/envs/gcar/bin/python` | `bev_fusion/tools/run_ar.sh`, `bev_fusion/project.json` (`Training.check`, `prepare`), `GtGenerator/run_nice.sh`, the lane's `tools/env/_py14.sh`, 13 READMEs; `jobrunner` takes it as an option (`--python`, `JOBRUNNER_PYTHON`) | make the environment under the same path (or a link to it): no file changes |
| 3 | the size of the machine: `gpu_slots` 2, `gpu_settle_s` 180, `ram_mb` 10000, `Training.gpu_max_mb` 3000 | `bev_fusion/project.json` (in git, so the same for every machine) | every one is an option of `run.py` (`--gpu-slots`, `--settle-s`, `--ram-mb`, `--gpu-max-mb`); a second machine with other sizes is the moment to read them from a file per machine in `state/` (not in git) |
| 4 | one GPU: a slot is a count, not a device; the memory gate reads the busiest GPU; nothing sets `CUDA_VISIBLE_DEVICES` | `tools/run.py` (`GpuSlot`, `gpu_used_mb`) | a box with several GPUs: slot i gets device i, the memory gate reads that device (about 20 lines and a test with a stand-in `nvidia-smi`) |
| 5 | "is it alive" is a pid of THIS host: the line of waiting launchers, the GPU slots, a command whose launcher died | `tools/run.py` (`Ticket`, `GpuSlot.stale`, `command_alive`), read by Training and jobrunner | one host per checkout: nothing to do. Several hosts on one checkout: the owner files get the host name, a holder of another host is judged by its heartbeat |
| 6 | the machine lock is one file per checkout | `state/machine_lock.json` (`tools/machine_lock.py`) | several hosts on one checkout: one lock file per host |
| 7 | the data: `~/data/nuscenes_full`, the stores on `D:`, the links of each problem | `data/NuScenes/README.md`, `data/NuScenes/tools/link_storage.sh --check`, `manifest.json` (names the raw data by hashes) | copy the data, run the check, make the links; the GT store is made again by `GtGenerator` from its tag |
| 8 | every run commits and tags the code | `tools/run.py` (`git_snapshot`) | the host that runs needs a git identity and write access to the repository; the tags are pushed from there |
| 9 | Windows-only: the report builders (PowerShell, pptx), mail through the browser | the problems' `reports/build/`, the loop prompt | stay on the workstation |

## Three ways to use a server, cheapest first
1. **One Linux box, the whole checkout on it, sessions started there.** Rows 1, 2, 7, 8. No code change; the hook line is the one decision.
2. **That box has several GPUs.** Row 4 in addition, then `gpu_slots` = number of GPUs.
3. **A cluster with a scheduler (Slurm or the like).** The scheduler owns the GPUs: a job is `run.py ... -- <command>` WITHOUT `--gpu`, one job file of `jobrunner` per submission; `--after` becomes the scheduler's dependency (inside an allocation it would wait and hold the GPU). Rows 5 and 6 when the nodes share one checkout. A submit command per scheduler is the only new code, and it belongs next to `jobrunner`, not into `run.py`.

## Where the tools reach into a suite (the boundaries to keep in mind)
`tools/` is problem-independent and standard library, but four places know a suite: `experiment.py` reads the kill rules with `Training.kill`, `run_report.py` draws with `Training.curves`, and `provenance.py`, `lineage.py` and `package.py gt` read the version list and the store layout of `GtGenerator`. All are imported inside the function that needs them. A second ground-truth generator or a second trainer is the moment to turn them into an entry in `project.json`; today it would be a layer for one user.
