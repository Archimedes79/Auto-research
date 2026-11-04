"""scratch: stage A, second round: the resume rules live in the config class; check_resume is the engine's"""
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8", newline="").read()


def cut(src, a, b, new=""):
    i, j = src.index(a), src.index(b)
    assert i < j, (a, b)
    return src[:i] + new + src[j:]


# the module-level lists after the class: gone (they stand before it now)
src = cut(src, "# a resume may change these on top of the engine's (paths, workers, the cadence of logging and saving, the switches that only look); every other difference is refused\nRESUME_FREE = _ENGINE_RESUME_FREE",
          "# ------------------------------------------------------------------------------------------------ glue")
# imports
src = src.replace("from trainkit.amp import AMP_DTYPES, amp_context, make_scaler, scaler_extra  # noqa: E402,F401 - the old names stay importable as train.X\n"
                  "from trainkit.checkpoint import prune_checkpoints, refuse_slim_resume, warm_start  # noqa: E402,F401\n"
                  "from trainkit.config import RESUME_FREE as _ENGINE_RESUME_FREE, Config  # noqa: E402\n",
                  "from trainkit.amp import amp_context, make_scaler, scaler_extra  # noqa: E402,F401 - the old names stay importable as train.X\n"
                  "from trainkit.checkpoint import check_resume, prune_checkpoints, refuse_slim_resume, warm_start  # noqa: E402,F401\n"
                  "from trainkit.config import Config  # noqa: E402\n")
src = src.replace("from trainkit.optim import (bad_steps_after, clip_gradients, clip_groups, lr_group_params, optimizer_step, param_groups,  # noqa: E402,F401\n"
                  "                            poisoned_buffers, retag_lr_groups)\n",
                  "from trainkit.optim import (bad_steps_after, clip_gradients, clip_groups, lr_group_params, optimizer_step, param_groups,  # noqa: E402,F401\n"
                  "                            poisoned_buffers)\n")
src = src.replace("import copy\n", "")
# the BEV wrapper of check_resume is gone
src = cut(src, "def check_resume(old, cfg, name):\n    \"\"\"trainkit.checkpoint.check_resume with the BEV lists", "# ------------------------------------------------------------------------------------------------ loop")
src = src.replace("tk_ckpt.resume(out_dir, model, loss, opt, ema, scaler, cfg, external=_external(model), free=RESUME_FREE, switches=RESUME_SWITCHES, override_env=\"BF_RESUME_ANY\")",
                  "tk_ckpt.resume(out_dir, model, loss, opt, ema, scaler, cfg, external=_external(model))")
open(path, "w", encoding="utf-8", newline="").write(src)
print("ok")
