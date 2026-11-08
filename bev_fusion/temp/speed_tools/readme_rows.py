"""throw-away: my README rows, written into the working-tree files (others' uncommitted lines stay; my earlier rows are replaced) and into the HEAD version of each file (-> temp/blob_*.md for commit_paths --blob)"""
import subprocess

REPO = "/mnt/c/Auto-research"
T_ROW = (
    "| `test_speed_flags.py` | the speed changes of `training/train.py` / `probe.py` that need no training run (CPU, seconds, toy data): the REAL `run_clip` against the per-term `float()` version kept in the file "
    "(16 cases: NaN at keyframe 0 / 1 / 2, bptt, 1 and 3 keyframes, bf16 and float64 terms; same terms, order, totals, gradients, Kendall values), the opt-in switches (`pin_memory`, `prefetch_factor`, "
    "`cudnn_benchmark`, `tf32`, `profile_steps`) off by default, `to_device(non_blocking)`, the resume rules of the new fields (a checkpoint from before the switches refuses `tf32` / `cudnn_benchmark`), "
    "`StepProfiler`: `profile.txt` once, a window of at most 8 steps, a failing report is printed and the profiler left | 9 planted bugs killed (float32 narrowing, no mean, no totals, early return, resume hole, "
    "no cap, raising report, profiler left entered, item twice; scratch `temp/speed_tools/mutate_host.py`) | passes (see the run) |\n"
    "| `test_host_overhead.py` | three cuts of host work against the code they replaced, kept in the file as the reference (CPU, seconds): `build_targets` computes the peak in its window only = the whole-grid "
    "version on ~1000 random boxes (edges, outside, radius 0.05 - 60 m, max_obj cap), `object_head_terms` reads the weight map with ONE slice (no GPU read per box) and gives the same targets and "
    "terms, `EMA.update` with multi-tensor calls = the per-tensor loop (frozen keys, integer buffers, bf16 / fp16 / fp64 state, every key frozen) | 13 planted bugs (window off by one / swapped / overwrite, read per "
    "box again, wrong sample, EMA decay in both paths / frozen keys / not decayed / bf16 through the multi-tensor call / no empty guard) killed, the baseline passes (scratch `temp/speed_tools/mutate_host.py`) | passes (see the run) |\n"
)
EDITS = {
    "bev_fusion/training/README.md": [
        ("schedule, gt.json written |",
         "schedule, gt.json written. Host-work cuts, default behaviour bit-identical (`../tests/test_speed_flags.py`, `../tests/test_host_overhead.py`): `run_clip` reads its logged terms back with one "
         "transfer per clip, `EMA.update` uses multi-tensor calls (float32 / float64 state); opt-in `pin_memory`, `prefetch_factor`, `cudnn_benchmark`, `tf32`; `profile_steps` n (capped at 8: 30 steps held 22 GB of host RAM) writes "
         "`profile.txt` into the run folder (GPU busy share, synchronisations per step, top operators) |"),
        ("Not yet wired into `train.py` (TrainConfig fields `train_only` / `freeze` and the optimiser from `trainable`: four hunks, waiting for the owner of `train.py`); test",
         "Wired into `train.py` (TrainConfig fields `train_only` / `freeze`, the optimiser from `trainable`); `StepProfiler` is the profile window of `profile_steps`; test"),
    ],
    "bev_fusion/tests/README.md": [("| `test_gt_prepare.py` |", T_ROW + "| `test_gt_prepare.py` |")],
}


def apply(text, edits):
    for old, new in edits:
        assert text.count(old) == 1, (text.count(old), old[:60])
        text = text.replace(old, new)
    return text


for path, edits in EDITS.items():
    head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, cwd=REPO).stdout.decode("utf-8")
    open(f"{REPO}/bev_fusion/temp/blob_{path.split('/')[-2]}_README.md", "w", encoding="utf-8", newline="").write(apply(head, edits))
    cur = open(f"{REPO}/{path}", encoding="utf-8", newline="").read()
    if path.endswith("tests/README.md"):                               # my rows are in already (first version): replace the slice up to the row after them
        a, b = cur.index("| `test_speed_flags.py` |"), cur.index("| `test_gt_prepare.py` |")
        cur = cur[:a] + cur[b:]
    elif "Host-work cuts" in cur:                                      # the sentence of the first version
        a = cur.index("schedule, gt.json written. Host-work cuts")
        b = cur.index("|", cur.index("`profile.txt` into the run folder", a))
        cur = cur[:a] + "schedule, gt.json written |" + cur[b + 1:]
        cur = cur.replace("Wired into `train.py` (TrainConfig fields `train_only` / `freeze`, the optimiser from `trainable`); `StepProfiler` is the profile window of `profile_steps`; test",
                          "Not yet wired into `train.py` (TrainConfig fields `train_only` / `freeze` and the optimiser from `trainable`: four hunks, waiting for the owner of `train.py`); test")
    open(f"{REPO}/{path}", "w", encoding="utf-8", newline="").write(apply(cur, edits))
    print("done", path)
