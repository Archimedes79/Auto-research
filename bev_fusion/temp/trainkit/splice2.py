"""scratch: stage A edits of train.py: cut the helpers that moved into autoresearch/trainkit (from marker to marker, the end marker stays)"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8", newline="").read()

OPT_NEW = '''# ------------------------------------------------------------------------------------------------ optimisation
# EMA, the schedule, the optimiser groups and step, AMP, warm start, pruning and the log files are the engine's (autoresearch/trainkit, imported above under the old names);
# what stays here names BEV things: the state that is not saved (the frozen camera heads) and the config fields a resume may change
def camera_frozen(model):
    """True if the camera heads are frozen (they come from the pre-trained file); a model without that switch counts as frozen"""
    return bool(getattr(getattr(model, "cfg", None), "freeze_camera", True))


def _external(model):
    """the state prefixes that are not in a checkpoint: the frozen camera heads. With freeze_camera false the camera weights are trained, so they belong to the state: a resume must
    restore them, not start them from the pre-trained file again"""
    return ("camera.",) if camera_frozen(model) else ()


def trainable_state(model, loss):
    """Everything that is not the frozen camera heads (those come from the pre-trained file)."""
    return tk_ckpt.model_state(model, loss, _external(model))


def save_checkpoint(path, model, loss, opt, ema, step, cfg, extra=None, slim=False):
    tk_ckpt.save_checkpoint(path, model, loss, opt, ema, step, cfg, extra, slim, external=_external(model))


def load_checkpoint(path, model, loss, opt, ema, ck=None):
    return tk_ckpt.load_checkpoint(path, model, loss, opt, ema, ck, external=_external(model))


def check_resume(old, cfg, name):
    """trainkit.checkpoint.check_resume with the BEV lists (RESUME_FREE, RESUME_SWITCHES); BF_RESUME_ANY=1 overrides the refusal"""
    tk_ckpt.check_resume(old, cfg, name, RESUME_FREE, RESUME_SWITCHES, "BF_RESUME_ANY")


'''

STOP_NEW = '''def _stop_requested(run_dir):
    """reason (a string) when the file CONTROL of the run directory says {"cmd": "stop"}, else None (`python -m trainsuite stop <run>`; the protocol is trainkit.control).
    The loop asks through this name, so a test can let a stop request arrive at a chosen step"""
    return control.stop_requested(run_dir)


'''


def cut(src, a, b, new=""):
    i, j = src.index(a), src.index(b)
    assert i < j, (a, b)
    return src[:i] + new + src[j:]


src = cut(src, "# ------------------------------------------------------------------------------------------------ optimisation\n", "# ------------------------------------------------------------------------------------------------ loop\n", OPT_NEW)
src = cut(src, "def detach_tree(x):", "def clip_step(prev_batch, device):")
src = cut(src, "AMP_DTYPES = {", "_CONTRACT_CHECKED = []")
src = cut(src, "EXIT_STOPPED = 75", "def record_gt(out_dir, datasets):", STOP_NEW)
open(path, "w", encoding="utf-8", newline="").write(src)
print("ok")
