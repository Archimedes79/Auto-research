import json
files = ["training/interface.py", "training/precompute.py", "training/train.py", "training/nuscenes_dataset.py", "training/feature_cache.py", "training/augment.py", "training/freeze.py",
         "training/probe.py", "training/debug.py", "training/scratch.py", "training/startable.py", "training/gt_cache_build.py", "training/gt_scenes.py", "training/gt_bev.py",
         "evaluation/bev_suite.py", "evaluation/bev_adapter.py", "evaluation/bev_task.py", "evaluation/metrics.py", "evaluation/bev_curves.py", "evaluation/bev_velocity.py",
         "evaluation/bev_fusion_area.py", "evaluation/bev_modules.py", "evaluation/bev_timing.py", "runner/bev_runner.py"]
json.dump({"about": "the frozen frame of the BEV system (FROZEN.md): path relative to bev_fusion/ -> SHA-256 (line endings ignored); rewritten only by tests/test_frozen.py --update", "frozen_at": "2026-11-02",
           "files": {f: "" for f in files}}, open("bev_fusion/frozen.json", "w"), indent=1)
