"""scratch: second round of my README rows (the critic's fixes): applied to the working file AND to the HEAD-based copy made by readme_edit.py"""
EDITS = {
    "bev_fusion/tests/README.md": ("bev_fusion/temp/t22/blob_tests_README.md", [
        ("the test removes its temporary folder (291 MB) when it ends. |",
         "critic round (2026-11-07, fresh `ar-critic`, no blocker): a one-frame batch (the stream step) of an 'all' model needs no `lane_img_old` and equals 'current'; the real ResNet-18 student makes the older frames "
         "one batch = one by one; 'all' in training mode (lift recomputed, older frames constants) equals the oracle bit for bit in output, gradients and state; the evaluation's camera controls "
         "(`bev_adapter.perturb_camera`, `bev_modules.zero_cam`, `bev_probe.perturb`) take `lane_img_old` away too, an unknown `lane_img*` item key is refused; a missing sweep image names `build-cache-older`; "
         "`BUILT.json` (one entry per n_cam / cam_stride, merged by a second build); `older_records` + the current frame = `feature_cache.used_frames`; `lane_images_older` defaults to off; "
         "15 more planted bugs killed (scratch `temp/t22/mut_K2.py`); the test removes its temporary folder (291 MB) when it ends. |"),
    ]),
    "bev_fusion/training/README.md": ("bev_fusion/temp/t22/blob_training_README.md", [
        ("read as uint8 `lane_img_old`;",
         "read as uint8 `lane_img_old`; a finished build writes `sweeps/BUILT.json` (one entry per n_cam / cam_stride; `startable.py` asks for it, so a half-built cache is not started on; a split other than the config's own scenes needs its own build);"),
    ]),
}
for path, (blob, edits) in EDITS.items():
    for target in (path, blob):
        s = open(target, encoding="utf-8", newline="").read()
        for old, new in edits:
            assert s.count(old) == 1, (target, old[:50], s.count(old))
            s = s.replace(old, new)
        open(target, "w", encoding="utf-8", newline="").write(s)
    print("edited", path, "and", blob)
