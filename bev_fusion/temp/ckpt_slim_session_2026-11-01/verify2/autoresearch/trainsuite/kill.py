"""Kill rules: the "kill criterion" of an experiment card as a line a program can check.

    group.lines@1000 > 1.48            the mean of the last 10 logged values of `group.lines` up to step 1000 is above 1.48 -> kill
    total@500 > 30 window 5            same with the last 5 values
    val.total@2000 >= 13.5             a validation row: the value of the eval row at step 2000 (or the first one after it)
A `#` starts a comment (in `kill.txt` and in `--kill`). Rules come from `--kill "..."` (repeatable) and from `<run>/kill.txt` (one rule per
line). A rule is `pending` until the run has reached its step, then `ok` or `hit`; `unknown` when the run passed the step and never logged
the key (a typo); a non-finite value is a `hit`. Nothing is stopped unless `watch --enforce` or `stop` is used.
"""
import math
import re

_RULE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*@\s*(\d+)\s*(>=|<=|>|<)\s*([-+0-9.eE]+)\s*(?:window\s+(\d+))?\s*$")
DEFAULT_WINDOW = 10


class Rule:
    def __init__(self, text):
        text = text.split("#", 1)[0]
        m = _RULE.match(text)
        if not m:
            raise ValueError(f"kill rule not understood: {text!r} (form: key@step > value [window N])")
        self.text = text.strip()
        self.key, self.step, self.op = m.group(1), int(m.group(2)), m.group(3)
        self.value, self.window = float(m.group(4)), int(m.group(5) or DEFAULT_WINDOW)

    def evaluate(self, run):
        """(status, measured value or None): status is 'pending', 'ok', 'hit' or 'unknown' (the key was never logged)"""
        if self.key.startswith("val."):
            reached = any(r["step"] >= self.step for r in run.evals)
            later = [v for s, v in run.series(self.key) if s >= self.step]
            x = later[0] if later else None
        else:
            reached = run.step >= self.step
            x = run.window_mean(self.key, self.step, self.window) if reached else None
        if x is None:
            return ("unknown" if reached else "pending"), None
        if not math.isfinite(x):
            return "hit", x
        hit = {">": x > self.value, ">=": x >= self.value, "<": x < self.value, "<=": x <= self.value}[self.op]
        return ("hit" if hit else "ok"), x


def load_rules(run, extra=()):
    """(rules, problems): problems are strings 'kill.txt line N: ...' / '--kill: ...' for lines that are not rules (the rest still works)"""
    rules, problems = [], []
    try:
        with open(run.p("kill.txt"), encoding="utf-8") as f:
            for n, ln in enumerate(f, 1):
                if ln.split("#", 1)[0].strip():
                    try:
                        rules.append(Rule(ln))
                    except ValueError as e:
                        problems.append(f"kill.txt line {n}: {e}")
    except OSError:
        pass
    for t in extra:
        try:
            rules.append(Rule(t))
        except ValueError as e:
            problems.append(f"--kill: {e}")
    return rules, problems
