p = "training/README.md"
L = open(p, encoding="utf-8").read().split("\n")
i = [k for k, ln in enumerate(L) if ln.startswith("| `probe.py` |")]
assert len(i) == 1
L[i[0] + 1:i[0] + 1] = [
    "| `startable.py` | is this config startable? the table of the keys of each design (`DESIGNS`), type and range checks, the cached inputs, the run folder's earlier state (a finished run is refused, a resume must have the same config), disk, the uncommitted files that go into the snapshot, the smoke twin (adapted from `bev_fusion/training/startable.py`); self-test | `python -m trainsuite start <config>` runs it first |",
    "| `configs/` | one run = one json file, presets of overrides (`presets/smoke.json`, `seed1.json`); see its [README](configs/README.md) | `configs/e271_grid.json` |",
    "| `train/run_config.py` | the command `trainsuite start` wraps: turns a config into the trainer's command line (design `grid_head` -> `grid_train.py`) | self-test |",
]
open(p, "w", encoding="utf-8").write("\n".join(L))
print("ok")
