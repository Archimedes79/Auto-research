"""Batch 4 of the structure review: README claims that contradict the repository, and the .gitignore block of problem 2."""
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BF = f"{ROOT}/bev_fusion"


def edit(path, old, new):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == 1, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8", newline="").write(s.replace(old, new))
    print("edited", os.path.relpath(path, ROOT))


edit(f"{BF}/README.md",
     "# bev_fusion - discussion folder: camera + radar BEV system\n\nFor now this folder only holds the discussion of a new BEV system (camera pixel-group cones, a learned radar density per detection, temporal fusion, semantic head) with Mirko. It is not yet a problem folder of `../autoresearch/` (no ledger, no runs); that comes when the plan is agreed.",
     "# bev_fusion - camera + radar BEV system\n\nThe second problem of this repository: a camera + radar BEV perception system (lift with depth weight, radar raster, temporal fusion with ego motion and a world memory, semantic / object / motion heads at any dt) on nuScenes. Plan and decisions in `plan.md`; code in `training/`; ground truth in `autogt/`; evaluation in `evaluation/` (the BEV adapter of `../autoresearch/evalsuite/`); ~30 trained runs in `runs/` (`t1`-`t4`). It is NOT yet managed by the process of `../autoresearch/` (no `project.json`, no ledger, no `run.py`): runs are launched by `temp/train/run_cfg.sh` from a pinned commit and compared by hand-written cards in `experiments/` (review of 2026-10-30: see `temp/review/`).")
edit(f"{BF}/README.md", "the papers of the scans of 2026-10-25 (lift, radar, temporal, heads, SDF, polar radar; 68 notes), moved here from `../bev_fusion_lane_center/literature/`",
     "the papers of the scans of 2026-10-25 and later (lift, radar, temporal, heads, SDF, polar radar; about 216 notes), partly moved here from `../bev_fusion_lane_center/literature/` - see `literature/README.md`")
edit(f"{BF}/README.md", "| `temp/setup/` | one-off scripts of this discussion (calibration check, table merges, the move out of the lane project) |",
     "| `temp/` | scripts of the studies and of the setup; some of them are used as launchers and recipes (`temp/train/run_cfg.sh`, the cache builders of `temp/setup/`) - see `temp/README.md` |\n| `runs/` | one folder per training run (not in git) - see `runs/README.md` |\n| `_to_delete/` | what was taken out of the live tree, listed in `_to_delete/README.md` |")
edit(f"{BF}/evaluation/README.md",
     "DEPRECATED forwarding stub of the old command line (running lane `temp/study/lane3.sh` calls it); delete when that lane has ended",
     "DEPRECATED forwarding stub of the old command line; no script calls it any more (checked 2026-10-30: `temp/study/lane3.sh` calls `bev_suite.py`; `temp/calib/prior_detector.py`, `temp/study/collect.py`, `build_compare_images.py` only name it in docstrings); delete it")

# .gitignore: problem 2 had no whitelist (runs/ with 30 code snapshots, data/ caches and the pre-trained weights showed up as untracked)
edit(f"{ROOT}/.gitignore",
     "# problem 2: camera + radar BEV (for now a discussion folder: plan, literature, one-off scripts)\n!/bev_fusion/\n",
     "# problem 2: camera + radar BEV: code and text records; runs (checkpoints, run snapshots, logs), caches and weights stay out\n"
     "!/bev_fusion/\n"
     "/bev_fusion/runs/*\n!/bev_fusion/runs/README.md\n"
     "/bev_fusion/data/*\n!/bev_fusion/data/README.md\n!/bev_fusion/data/scenes_parts1_6.json\n"
     "/bev_fusion/models/pretrained/*\n!/bev_fusion/models/pretrained/*.sha256\n")
