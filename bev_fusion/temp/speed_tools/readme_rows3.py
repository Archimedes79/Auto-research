"""throw-away: the README rows of the hygiene flag and lr_groups, applied to the working-tree files (others' uncommitted lines stay) and to the HEAD version of each file (-> temp/blob3_* for commit_paths --blob)"""
import subprocess

REPO = "/mnt/c/Auto-research"
T_ROWS = (
    "| `test_hygiene.py` | `LossConfig.hygiene_2026_11_02` and the splat precision (records/open_findings_2026-11-02.md #1 - #4, #6; CPU, seconds): neutral inputs (weights 1, velocities known) give the flag off bit for bit; "
    "`w_obj` is a weight (the regression terms are weighted means of single-object runs, a focal cell of weight 0.5 gets half the gradient, the motion L1 weighted by hand); an unknown-velocity object is out of the "
    "moving / static BCE and the velocity L1 and its logit gets no gradient; the dead `radar_slot` branch is gone and an old config with `w_radar_slot` loads; the flag reaches the object head and the motion loss "
    "through `BEVLoss` and `train.forward_loss`; bf16 splat geometry is float32 (0.4 m cells, ego chains); `batch_ego(strict=)` | 23 planted bugs killed (scratch `temp/speed_tools/mutate_host.py`) | passes (see the run) |\n"
    "| `test_lr_groups.py` | `TrainConfig.lr_groups` (experiments/t17_motion_lr section 3; CPU, seconds): without it exactly today's two groups; with a prefix four groups tagged `base_lr` / `warmup` / `lr_group` "
    "(`motion_blockx` and the loss parameters stay in the rest); the schedule against the formula by hand (lr 1e-4, warm-up 1000: 3.34e-5 at step 0, 1e-4 at 999 / 1000); every bad value an error; the REAL "
    "`train()` on a toy: optimiser groups, `lr.<prefix>` columns, a tiny group lr hardly moves its weights, a typo prefix fails before the loaders are built, a resume with other lr_groups or on a checkpoint "
    "from before the field is refused | 11 planted bugs killed (same scratch tool) | passes (see the run) |\n"
    "| `toy_train.py` | helper, not a test: the real `train()` on a toy network on the CPU (data path stubbed), used by `test_lr_groups.py` | - | - |\n"
)
EDITS = {
    "bev_fusion/training/README.md": [
        ("the scaler state is in the checkpoint's `extra`; `../tests/test_grad_scaler.py`) |",
         "the scaler state is in the checkpoint's `extra`; `../tests/test_grad_scaler.py`). `lr_groups` {module prefix: {lr, warmup}} gives a module its own learning rate and warm-up (`lr_group_params`, "
         "`param_groups`, `scheduled_lr`; {} = today's two groups, `../tests/test_lr_groups.py`). `LossConfig.hygiene_2026_11_02` (off = bit for bit): `w_obj` weights the object head and the motion loss, "
         "unknown-velocity objects leave the moving / static head alone, `batch_ego` raises on other dts (`../tests/test_hygiene.py`); `splat.py` computes positions in float32 under bf16 / fp16 |"),
    ],
    "bev_fusion/tests/README.md": [("| `test_gt_prepare.py` |", T_ROWS + "| `test_gt_prepare.py` |")],
    "bev_fusion/tests/run_gates.sh": [
        ('run "grad scaler"       "$PY" tests/test_grad_scaler.py            # amp fp16: scaler exact against plain steps, overflow, bad-step rule, checkpoint',
         'run "grad scaler"       "$PY" tests/test_grad_scaler.py            # amp fp16: scaler exact against plain steps, overflow, bad-step rule, checkpoint\n'
         'run "lr groups"         "$PY" tests/test_lr_groups.py              # a module\'s own lr and warm-up: groups, schedule, resume\n'
         'run "hygiene"           "$PY" tests/test_hygiene.py                # w_obj as a weight, unknown velocity, splat precision, batch_ego (the flag hygiene_2026_11_02)'),
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
    open(f"{REPO}/bev_fusion/temp/blob3_{name}", "w", encoding="utf-8", newline="").write(apply(head, edits))
    continue                                                           # the working-tree files of a shared checkout can be older than HEAD: only the blob (HEAD + my rows) is written
    cur = open(f"{REPO}/{path}", encoding="utf-8", newline="").read()
    open(f"{REPO}/{path}", "w", encoding="utf-8", newline="").write(apply(cur, edits))
    print("done", path, "-> blob3_" + name)
