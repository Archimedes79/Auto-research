import re
p = "training/train.py"
s = open(p, encoding="utf-8").read()
old_start = s.index("def _stop_requested(run_dir, since):")
old_end = s.index("def train(cfg, eval_fn=None, datasets=None):")
new_funcs = '''def _stop_requested(run_dir):
    """reason (a string) when the file CONTROL of the run directory says {"cmd": "stop"}, else None. Same code as
    autoresearch/trainsuite/control.py (inlined: a run trains from a code snapshot that does not contain that package)."""
    try:
        with open(os.path.join(run_dir, "CONTROL"), encoding="utf-8") as fh:
            d = json.load(fh)
        return str(d.get("reason", "")) if d.get("cmd") == "stop" else None
    except (OSError, ValueError, AttributeError):
        return None


def _clear_control(run_dir):
    """at the start: move CONTROL / STOPPED of an earlier stop aside (*.old), so a new process never obeys an old request (no clocks)"""
    for n in ("CONTROL", "STOPPED"):
        p = os.path.join(run_dir, n)
        if os.path.exists(p):
            os.replace(p, p + ".old")


'''
s = s[:old_start] + new_funcs + s[old_end:]
s = s.replace("def train(cfg, eval_fn=None, datasets=None):\n    t_start = time.time()\n", "def train(cfg, eval_fn=None, datasets=None):\n")
s = s.replace("    os.makedirs(out_dir, exist_ok=True)\n    model, loss, _ = build(cfg)\n", "    os.makedirs(out_dir, exist_ok=True)\n    _clear_control(out_dir)\n    model, loss, _ = build(cfg)\n", 1)
s = s.replace("reason = _stop_requested(out_dir, t_start)", "reason = _stop_requested(out_dir)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
