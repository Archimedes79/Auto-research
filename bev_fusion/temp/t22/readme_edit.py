"""scratch: my rows of tests/README.md and training/README.md (student_frames 'all', section K, the encoder test's skip, the drift refusal): applied to the working files AND to HEAD copies
(temp/t22/blob_*.md) that hold only these edits, so the commit does not carry another chat's rows (test_ab_timing / ab_timing)."""
import subprocess

EDITS = {
    "bev_fusion/training/README.md": [
        ("read by `CachedFrontRadarDataset(lane_images=...)` as uint8 `lane_img`);",
         "read by `CachedFrontRadarDataset(lane_images=...)` as uint8 `lane_img`); the OLDER frames of a pair / window for `ModelConfig.student_frames` 'all' (t22): `build-cache-older` / `check-cache-older <config>` into "
         "`<lane_images>/sweeps/` (same crop, one file per CAM_FRONT sweep token; `older_records` takes exactly the frames the dataset reads), read as uint8 `lane_img_old`;"),
    ],
    "bev_fusion/tests/README.md": [
        ("`student_frames` 'all' = NotImplementedError,", "`student_frames` 'all' accepted with a student and refused without one,"),
        ("the bytes saved for the backward fall (8 planted bugs, 8 killed). |",
         "the bytes saved for the backward fall (8 planted bugs, 8 killed); the same equality for EfficientViT-B1, and the global random state is unchanged by the recomputation. "
         "K `student_frames` 'all' (2026-11-07; the older frames of the pair from their own images in `<lane_images>/sweeps`, `lane_img_old`): BEV = the student-free model's on the batch whose every frame is the "
         "student's output of that frame's own image (oracle made one frame at a time, F = 2, 3), frame 0 as with 'current', 'current' ignores `lane_img_old`, garbage in the cached older frames changes nothing, "
         "a missing or miscounted `lane_img_old` is an error, the older outputs carry no graph and the gradient reaches the student by the current frame only (also in training mode with the recomputed lift), "
         "`collate_cached` and the dataset item (independent JPEG decode), `older_records` / `_fill_older` (resumable), `student_findings`, the dataset wiring of `train.make_datasets`; all 20 planted bugs of K are killed "
         "(scratch `temp/t22/mut_K.py`, run on a private copy of the code); the test removes its temporary folder (291 MB) when it ends. |"),
        ("but it reads the ImageNet weight files of `models/pretrained/imagenet/` from the working directory)",
         "but it reads the ImageNet weight files of `models/pretrained/imagenet/` from the working directory; the EfficientViT-B1 file is not in git: without it the weight check of section 2 and the CLI runs of section 6 are skipped, the rest runs)"),
        ("D: a planted exact difference of 0.5 is refused by the same limit |",
         "D: a planted exact difference of 0.5 is refused by the same limit; B also: a `camera_student` chain is refused up front by `stream_drift.py` (no data needed) |"),
    ],
}
for path, edits in EDITS.items():
    work = open(path, encoding="utf-8", newline="").read()
    head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, check=True).stdout.decode("utf-8")
    for old, new in edits:
        assert work.count(old) == 1, (path, old[:60], work.count(old))
        assert head.count(old) == 1, ("HEAD", path, old[:60], head.count(old))
        work = work.replace(old, new)
        head = head.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(work)
    blob = "bev_fusion/temp/t22/blob_" + path.split("/")[-2] + "_README.md"
    open(blob, "w", encoding="utf-8", newline="").write(head)
    print("edited", path, "->", blob)
