"""The files a trainer keeps in its run directory besides the checkpoints: log.jsonl (one JSON row per logged step, `{"step", "val": {...}}` rows for the evaluations)."""
import json
import os


def supersede_log(out_dir, step):
    """on a resume from `step`: the rows of log.jsonl above it (the steps redone after a crash) move to log_superseded.jsonl, so the log
    has one row per step again (before 2026-10-31 a resumed run had steps 520 - 1000 twice)"""
    path = os.path.join(out_dir, "log.jsonl")
    if not os.path.exists(path):
        return
    keep, gone = [], []
    for ln in open(path, encoding="utf-8"):
        try:
            s = json.loads(ln).get("step", 0)
        except ValueError:                                     # a row cut by the crash
            s = step + 1
        (gone if s > step else keep).append(ln if ln.endswith("\n") else ln + "\n")
    if gone:
        with open(os.path.join(out_dir, "log_superseded.jsonl"), "a", encoding="utf-8") as f:
            f.writelines(gone)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            f.writelines(keep)
        try:
            os.replace(path + ".tmp", path)
        except OSError as e:                                   # a reader on the Windows side holds log.jsonl open: keep appending
            os.remove(path + ".tmp")
            print(f"WARNING: log.jsonl not rewritten ({type(e).__name__}: {e}); its rows above step {step} stay in it", flush=True)


def val_logged(out_dir, step):
    """True if log.jsonl has the validation row of `step`"""
    path = os.path.join(out_dir, "log.jsonl")
    if not os.path.exists(path):
        return False
    for ln in open(path, encoding="utf-8"):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("step") == step and "val" in r:
            return True
    return False
