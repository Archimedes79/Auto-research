# FROZEN - the frame of the BEV system is fixed, the work is in the modules (D44, Mirko 2026-11-02: "Freeze a part of the code, concentrate on the module code")

What is frozen is everything that makes a number comparable between two runs or that every module experiment stands on: the contracts between the layers, the data path, the trainer, the evaluation and the ground-truth tools. What stays open is the code of the modules, their losses and the model's builder tables.

| frozen (the frame) | open (the modules) |
|---|---|
| `training/interface.py` (the contract of what the network reads and returns), `precompute.py` (layer 1, the visibility maps), `train.py` (the trainer), `nuscenes_dataset.py`, `feature_cache.py`, `augment.py` (the data path), `freeze.py`, `probe.py`, `debug.py`, `scratch.py`, `startable.py`, `gt_cache_build.py`, `gt_scenes.py`, `gt_bev.py` | `camera_encoder.py`, `lift_pd.py`, `bevfusion_camera.py`, `radar_bev.py`, `radar_encoder.py`, `splat.py`, `temporal.py`, `temporal_fusion.py`, `motion_block.py`, `fusion.py`, `bev_trunk.py`, `blocks_new.py`, `fallbacks.py`, `object_head.py`, `timed_head.py`, `extract.py`, `loss_extras.py`, `lane_branch.py`, and `model.py` / `losses.py` (a new kind of a slot is one table entry with its test in `tests/test_slots.py`) |
| `evaluation/bev_suite.py`, `bev_adapter.py`, `bev_task.py`, `metrics.py`, `bev_curves.py`, `bev_velocity.py`, `bev_fusion_area.py`, `bev_modules.py`, `bev_timing.py` (what a number means) | `training/configs/` (a new config is a new file, never an edit of a stored one) |
| `runner/bev_runner.py`; `../autogt/` (it has its own version signature), `../autoresearch/` (its own rules) | experiments, cards, tests of the modules |

**The rule.** A frozen file changes only for a bug: a test that fails before the change and passes after it, one `ar-critic` pass (mode code), a line in the changelog below, and `python tests/test_frozen.py --update` (it writes the new hashes into `frozen.json`). A change that alters a number of an old run is not a bug fix: it is a new version of the frame and needs Mirko's word. The check runs with the gates (`tests/run_gates.sh`, row "frozen frame") and fails on every other edit of these files.

**Why.** Run comparisons of the next days (t17 data, t18 fixes, t19 radar memory, the seed twins) are only comparable when the trainer, the data path and the evaluation are the same bits; the baseline `t16.full_s0` trained from commit `40e7ff5b3c` (tag `bf/t16.full_s0`), the arms are pinned to it.

## Changelog
- 2026-11-02: frame frozen at the state after the hygiene flag and `lr_groups` (`df9d437`) and the split clip (`e1e28f2`); `frozen.json` holds the hashes of that state.
- 2026-11-02: `evaluation/bev_timing.py` idle guard (bug fix): a `run.py` launcher merely queued for a GPU slot, whose arguments name train.py / run_ar.sh / bev_suite, was counted as a GPU user and refused the timing on an idle GPU (0 % utilisation, 0.9 GB). Only a python process running a trainer / evaluation / probe script on the GPU counts now (`holds_gpu`); the self-test fails with the old logic (queued launchers), `ar-critic` pass done (interpreter options with a value, `-m` forms, the `modules` command, more GPU scripts added). No number of any run changes.
