"""throw-away: the README rows of the gradient scaler, applied to the working-tree files (others' uncommitted lines stay) and to the HEAD version of each file (-> temp/blob_*.md for commit_paths --blob)"""
import subprocess

REPO = "/mnt/c/Auto-research"
T_ROW = (
    "| `test_grad_scaler.py` | `amp = \"fp16\"` with a GradScaler (`training/train.py`: `amp_context`, `make_scaler`, `optimizer_step`, `bad_steps_after`, `scaler_extra`, `run_clip(scaler=)`; CPU, seconds, "
    "toy network): autocast dtype per name and an unknown name is an error; 5 steps with a scale of 1024 give the same weights and gradient norms as 5 plain steps bit for bit (a power of two is exact) "
    "and the probe hook sees unscaled gradients; inf / nan in a gradient and a gradient whose norm alone overflows skip the step, drop the gradients and halve the scale; an overflow counts as a bad "
    "step only below scale 1; `run_clip` gradients are exactly scale x the plain ones; the scaler state travels in the checkpoint; amp / fp16_scale are not free on a resume | 11 planted bugs "
    "(no unscale, scale not lowered, norm-only overflow steps, gradients kept, bad-step rule both ways, loss not scaled, wrong dtype, scaler for bf16, state not saved, unknown name accepted) "
    "killed, the baseline passes (scratch `temp/speed_tools/mutate_host.py`) | passes (see the run) |\n"
)
EDITS = {
    "bev_fusion/training/README.md": [
        ("`profile.txt` into the run folder (GPU busy share, synchronisations per step, top operators) |",
         "`profile.txt` into the run folder (GPU busy share, synchronisations per step, top operators). `amp = \"fp16\"` trains with a GradScaler (`make_scaler`, start scale `fp16_scale` 1024; "
         "`optimizer_step` unscales, clips and steps, an overflow skips the step and halves the scale; the scaler state is in the checkpoint's `extra`; `../tests/test_grad_scaler.py`) |"),
    ],
    "bev_fusion/tests/README.md": [("| `test_gt_prepare.py` |", T_ROW + "| `test_gt_prepare.py` |")],
    "bev_fusion/tests/run_gates.sh": [
        ('run "host overhead"     "$PY" tests/test_host_overhead.py          # target window, one host read per call, multi-tensor EMA',
         'run "host overhead"     "$PY" tests/test_host_overhead.py          # target window, one host read per call, multi-tensor EMA\n'
         'run "grad scaler"       "$PY" tests/test_grad_scaler.py            # amp fp16: scaler exact against plain steps, overflow, bad-step rule, checkpoint'),
    ],
}


def apply(text, edits):
    for old, new in edits:
        assert text.count(old) == 1, (text.count(old), old[:60])
        text = text.replace(old, new)
    return text


for path, edits in EDITS.items():
    head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, cwd=REPO).stdout.decode("utf-8")
    name = path.split("/")[-2] + "_" + path.split("/")[-1].replace(".", "_")
    open(f"{REPO}/bev_fusion/temp/blob2_{name}", "w", encoding="utf-8", newline="").write(apply(head, edits))
    cur = open(f"{REPO}/{path}", encoding="utf-8", newline="").read()
    open(f"{REPO}/{path}", "w", encoding="utf-8", newline="").write(apply(cur, edits))
    print("done", path, "-> blob2_" + name)
