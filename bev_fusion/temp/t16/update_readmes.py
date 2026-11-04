"""scratch (2026-11-02, Mirko: 'the readme in git does not reflect the current status'): the two READMEs from their HEAD text plus a dated status block and a quick start.
Run from the repository root: python3 bev_fusion/temp/t16/update_readmes.py  (writes README.md and bev_fusion/README.md; each replacement must match exactly once)"""
import subprocess


def head(path):
    return subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True, encoding="utf-8").stdout


def once(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:70])
    return s.replace(a, b)


# ---------------------------------------------------------------- bev_fusion/README.md
s = head("bev_fusion/README.md")
s = once(s, "trained runs in `runs/` (`t1`-`t10`)", "trained runs in `runs/` (`t1`-`t18` so far)")

status = """
## Status (2026-11-02; live numbers are in `records/LEDGER.md` and the cards, not here)
- **The framework.** The five layers (precompute, input interface, network slots, output interface, post-processing) are built and tested: `tests/run_gates.sh` (fast gates), `tests/test_slots.py` builds every kind of every slot and 13 combinations, `tests/equivalence.py` reproduces stored checkpoints. The ground truth is autogt v8 (`../data/autogt/c643a7539d`: all 850 trainval sequences, 34,149 keyframes; v5 is deleted, the versions are in `../autogt/README.md`).
- **The full network** (config `training/configs/t16_full.json`, card `experiments/t16_full_network/`): dual camera features (DualBEV + BEVFusion, cached), delta-slot radar encoder, camera memory, concat fusion, DualBEV trunk, map / object / velocity heads with the velocity hurdle, the motion block in its joint form (the box velocity is read from its field) and a separate gradient clip for it. Brought up on autogt's small set (10 scenes): smoke test, 400-step runs, a module probe (every module's output is finite, not constant and responds to its input), timing (96.4 ms per frame at batch 1 on an idle RTX 3060, 12 % above the old chain; 3.6 s per training step at accumulation 2). Eight findings and their fixes are in the card (section 6); the radar memory with the Doppler static gate is off in the large run (an unlearned block).
- **Running or queued** (state of 2026-11-02): the large baseline `t16.full_s0` (6000 steps, 428 train / 82 val scenes) and its matched control `t16.ctl_s0`; the data arm `t17.data` (700 train scenes, `experiments/t17_more_data/`) and the bundle arm `t18.fix` (the Training chat's hygiene fixes + the motion block's own learning rate, `experiments/t18_motion_lr/`). Their results go into the cards and the ledger; this README holds none of them.
- **Open:** `records/open_findings_2026-11-02.md` (trainer / loss findings of the bug hunt, ordered), `records/cleanup_inventory_2026-11-02.md` (what can go, by which rule), `plan/head_design.md` (the heads' open GT: centre lines, radar stationary target, horizon beyond 0.5 s).

## Quick start
From this folder, in the WSL (env `gcar`, `~/miniconda3/envs/gcar/bin/python`); the tools need `AR_PROJECT=../bev_fusion` (the default problem is the lane one).

| what | command |
|---|---|
| tests | `bash tests/run_gates.sh`; `python tests/test_slots.py` |
| check a config without a GPU | `python training/startable.py training/configs/t16_full.json` |
| start a training | from `../autoresearch/`: `python -m trainsuite start ../bev_fusion/training/configs/t16_full.json --id <id> --hyp "..." --pred "..." --rule "..." --go`; a small-set bring-up: `--preset small --preset small_set --no-card`; a chain: `--run-arg=--after --run-arg=<id>`; the launcher is `../autoresearch/tools/run.py`, the details are in `records/README.md` |
| watch it | from `../autoresearch/`: `python -m trainsuite --root ../bev_fusion/runs status <run>` (also `health`, `curves`, `queue`, `stop`) |
| evaluate a run | `python evaluation/bev_suite.py runs/<run> --curves --fusion --velocity eval --n 1000 --prior 500`; further commands: `... modules` (module probe), `... timing` (idle GPU), `... qual videos 3`, `... generalisation`, `... report --vs runs/<control>`, `compare` (paired bootstrap): see `evaluation/README.md` |
| the ground truth | `python -m autogt sets` (what the store holds), `python -m autogt generate ...`: `../autogt/README.md` |
| the system on frames without GT | `python runner/bev_runner.py precompute / network / full ...`: `runner/README.md` |
"""
s = once(s, "| file / folder | content |\n|---|---|\n", status.lstrip("\n") + "\n## Folders\n| file / folder | content |\n|---|---|\n")
s = once(s, "| `runs/` | one folder per training run (not in git) - see `runs/README.md` |",
         "| `data/` | derived data, not in git: the caches of the frozen image heads (`cache/cam_feats*`, big ones on D: behind links), the GT and radar caches of the trainer, the small-set links - see `data/README.md` |\n"
         "| `runs/` | one folder per training run (not in git) - see `runs/README.md` |")
s = once(s, "first: `s1a_lift_pd_vs_p/` (camera-only real-data run, its synthetic unit test, idle-GPU timing) - see `experiments/README.md`",
         "from `s1a_lift_pd_vs_p/` and `t1` to `t18`; the current ones: `t16_full_network/` (the full network and its bring-up log), `t17_more_data/`, `t18_motion_lr/` - see `experiments/README.md`")
s = once(s, "Shared data (nuScenes) is described in `../data/NuScenes/README.md`.", "Shared data (nuScenes) is described in `../data/NuScenes/README.md`; the rules for agents working here are in `../CLAUDE.md`.")
open("bev_fusion/README.md", "w", encoding="utf-8", newline="\n").write(s)

# ---------------------------------------------------------------- README.md (the front page)
r = head("README.md")
r = once(r, "| [`bev_fusion/`](bev_fusion/README.md) | problem 2: a camera + radar BEV perception system (training, evaluation, experiments, literature) |",
         "| [`bev_fusion/`](bev_fusion/README.md) | problem 2: a camera + radar BEV perception system on nuScenes: the five-layer framework and the full network (camera + radar + temporal fusion + motion block) on ground truth v8; training, evaluation, experiments, literature |")
status_root = """## Status (2026-11-02)
- **Method and tools** are in place and tested: the launcher with a git tag per run, ledger, decisions, the training suite (`status` / `health` / `stop` / `speed`), the evaluation suite (frames, videos, numbers, paired comparison, module probe, timing) and the ground-truth suite autogt (v8 release: all 850 trainval sequences, 34,149 keyframes).
- **Problem 2 (BEV system):** the five-layer framework and the full network are built; the small-set bring-up (timing, module probes, eight findings and fixes) is done; the large baseline run, its matched control and two improvement arms are training or queued. Details and the quick start: [`bev_fusion/README.md`](bev_fusion/README.md), the bring-up log: [`bev_fusion/experiments/t16_full_network/experiment.md`](bev_fusion/experiments/t16_full_network/experiment.md). Live numbers are in the ledgers and cards, not in the READMEs.
- **Problem 1 (lane centre-line graph):** [`bev_fusion_lane_center/README.md`](bev_fusion_lane_center/README.md) and its `HANDOVER.md`.
- **Data:** raw nuScenes on D:, caches and checkpoints next to it, reached through links; the map is [`data/NuScenes/README.md`](data/NuScenes/README.md).

"""
r = once(r, "## Where to start\n", status_root + "## Where to start\n")
open("README.md", "w", encoding="utf-8", newline="\n").write(r)
print("READMEs written")
