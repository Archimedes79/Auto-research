"""Kill rules: the "kill criterion" of an experiment card as a line a program can check.

    group.lines@1000 > 1.48            the mean of the last 10 logged values of `group.lines` up to step 1000 is above 1.48 -> kill
    total@500 > 30 window 5            same with the last 5 values
    val.total@2000 >= 13.5             a validation row: the value of the eval row at step 2000 (or the first one after it)
Rules come from `--kill "..."` (repeatable) and from `<run>/kill.txt` (one rule per line, `#` comments). A rule is `pending` until the run
has reached its step, then `ok` or `hit`. Nothing is stopped unless `watch --enforce` or `stop` is used.
"""
import os
import re

_RULE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*@\s*(\d+)\s*(>=|<=|>|<)\s*([-+0-9.eE]+)\s*(?:window\s+(\d+))?\s*$")
DEFAULT_WINDOW = 10


class Rule:
    def __init__(self, text):
        m = _RULE.match(text)
        if not m:
            raise ValueError(f"kill rule not understood: {text!r} (form: key@step > value [window N])")
        self.text = text.strip()
        self.key, self.step, self.op = m.group(1), int(m.group(2)), m.group(3)
        self.value, self.window = float(m.group(4)), int(m.group(5) or DEFAULT_WINDOW)

    def evaluate(self, run):
        """(status, measured value or None): status is 'pending', 'ok' or 'hit'"""
        if self.key.startswith("val."):
            later = [v for s, v in run.series(self.key) if s >= self.step]
            x = later[0] if later else None
        else:
            x = run.window_mean(self.key, self.step, self.window) if run.step >= self.step else None
        if x is None:
            return "pending", None
        hit = {">": x > self.value, ">=": x >= self.value, "<": x < self.value, "<=": x <= self.value}[self.op]
        return ("hit" if hit else "ok"), x


def load_rules(run, extra=()):
    rules = []
    try:
        with open(run.p("kill.txt"), encoding="utf-8") as f:
            rules += [Rule(ln) for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    except OSError:
        pass
    return rules + [Rule(t) for t in extra]
