# scratch: the changelog line of records/frame_changes.md for the removal of student_frames 'all' (the lead's order 2026-11-10)
LINE = ("- 2026-11-10 (clean-base round, Image Encoder chat, commit e7366b23ec): the t22 variant `ModelConfig.student_frames` 'all' (stage B2, CUT) is removed from the live code "
        "(`training/lane_branch.py` older-frame image cache, `feature_cache.py`, `interface.py`, `startable.py`, `train.py`, `model.py`: the field stays with the single value 'current'; "
        "the evaluation's `lane_img_old` keys stay, they are the Evaluation chat's), **no number of any run changes**. Stored runs with 'all': `runs/t22_B2_s0` and `runs/smoke_t22_B2_s0` "
        "evaluate from their own `code/` snapshot; `runs/t22_B_all`, `runs/t22_standin_all_pair4` and `runs/smoke_t22_all_cpu` had no snapshot and imported the live `training/`, so each got a pinned "
        "`code/` copy of the tag `bf/t22.B_all` / `bf/t22.standin_all_pair4` (`training/` + `trainkit/`; `code/SNAPSHOT.txt` says where it came from; nothing else in the run folder changed). "
        "Checked on the CPU through `bev_adapter.load_run` (a stand-in run keeps its weights in `runs/t22_B_s0` / `runs/t21_pair4_s0`: `--ckpt-dir`): all seven stored t22 runs load, and "
        "`perturb_camera`, `bev_modules.zero_cam` and `bev_probe.perturb` zero or shuffle `lane_img_old` for each of them. The code of the variant: commit `1648bcc24d`, tag `bf/t22.B2_s0`.\n")
EDITS = [
    ("bev_fusion/records/frame_changes.md", r"^## Changelog\n", "## Changelog\n" + LINE),
]
