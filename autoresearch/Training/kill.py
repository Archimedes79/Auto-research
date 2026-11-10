"""Kill rules: the "kill criterion" of an experiment card as a line a program can check.

    group.lines@1000 > 1.48            the mean of the last 10 logged values of `group.lines` up to step 1000 is above 1.48 -> kill
    total@500 > 30 window 5            same with the last 5 values
    val.total@2000 >= 13.5             a validation row: the value of the eval row at step 2000 (or the first one after it)
A `#` starts a comment (in `kill.txt` and in `--kill`). Rules come from `--kill "..."` (repeatable) and from `<run>/kill.txt` (one rule per
line). A rule is `pending` until the run has reached its step, then `ok` or `hit`; `unknown` when the run passed the step and never logged
the key (a typo); a non-finite value is a `hit`. Nothing is stopped unless `watch --enforce` or `stop` is used.
"""
import codecs
import math
import re

_RULE = re.compile(r"^\s*([A-Za-z0-9_.\-@+/]+?)\s*@\s*(\d+)\s*(>=|<=|>|<)\s*([-+0-9.eE]+)\s*(?:window\s+(\d+))?\s*$")
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


def read_text(path):
    """the text of a rule file: UTF-8 (a BOM is fine) or UTF-16 with a BOM, which is what PowerShell 5.1 (`>`, `Out-File`) writes; raises OSError for a
    missing file and UnicodeDecodeError for bytes that are neither"""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_rules(run, extra=()):
    """(rules, problems): problems are strings 'kill.txt line N: ...' / '--kill: ...' for lines that are not rules (the rest still works); a kill.txt that is
    no readable text is one problem, never an exception (one bad file must not stop the status of all runs)"""
    rules, problems = [], []
    try:
        text = read_text(run.p("kill.txt"))
    except OSError:
        text = ""
    except UnicodeDecodeError as e:
        text = ""
        problems.append(f"kill.txt is not text in UTF-8 or UTF-16 with a BOM ({e.reason} at byte {e.start}; saved as ANSI / cp1252? save it as UTF-8): none of its rules is used")
    for n, ln in enumerate(re.split(r"\r\n|\r|\n", text), 1):       # not splitlines(): it also breaks at form feed, U+2028 ... and would shift the line numbers
        if ln.split("#", 1)[0].strip():
            try:
                rules.append(Rule(ln))
            except ValueError as e:
                problems.append(f"kill.txt line {n}: {e}")
    for t in extra:
        try:
            rules.append(Rule(t))
        except ValueError as e:
            problems.append(f"--kill: {e}")
    return rules, problems
