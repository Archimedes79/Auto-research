"""Reading one training run directory (standard library only).

A run directory follows the contract of README.md: `config.json` (key `steps`), the metrics stream `log.jsonl`, the text log `train.log`,
the flag `.done` (only for exit code 0), the control files `CONTROL` / `STOPPED`. Layouts that differ are described in an optional
`trainsuite.json` inside the run directory: {"log": "metrics.jsonl", "text_log": "run.log", "loss": "loss", "steps": 3000, "progress": "it "}.
"""
import json
import math
import os
import re
import statistics
import time

STALE_S = float(os.environ.get("TS_STALE_S", "900"))      # no new output for this long and not finished = stalled
STOPPED_GRACE_S = 10.0                                    # output this much newer than the STOPPED marker: the run went on after the stop


def _flat(obj, prefix=""):
    """numeric leaves of a JSON record as {"a.b": float}; strings, lists and booleans are dropped"""
    out = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                out[f"{prefix}{k}"] = float(v)
            except OverflowError:                         # an integer beyond the float range (json.loads reads any number of digits): no value, not a crash of every status
                pass
    return out


def read_log(path, resumes=None):
    """(train rows, eval rows, restarts) of a metrics stream. A row is a flat dict with `step`; an eval row has keys `val.*`.
    After a resume the file holds steps twice: a row whose step is not above the last one replaces everything from that step on
    (the later attempt wins, its evaluations from that step on included); `restarts` counts those replacements. Contract: ONE train row per step
    (a second train-type row at the same step, e.g. a closing summary row, counts as a rollback and drops the evaluations from that step on).
    A half-written last line is skipped. A list given as `resumes` receives the step of the row that replaced the old ones, once per restart."""
    train, evals, restarts = [], [], 0
    try:
        f = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return train, evals, 0
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or "step" not in rec:
                continue
            row = _flat(rec)
            if "step" not in row or not math.isfinite(row["step"]):          # NaN / Infinity as a step (json.loads accepts them): no row, not a crash of every later int(step)
                continue
            lst = evals if isinstance(rec.get("val"), dict) else train          # by the record, not by its keys: an empty val dict is an eval row
            if lst and row["step"] <= lst[-1]["step"]:
                while lst and lst[-1]["step"] >= row["step"]:
                    lst.pop()
                if lst is train:
                    restarts += 1
                    if resumes is not None:
                        resumes.append(int(row["step"]))
                    while evals and evals[-1]["step"] >= row["step"]:        # the evaluations of the superseded training suffix go with it
                        evals.pop()
            lst.append(row)
    return train, evals, restarts


class Run:
    def __init__(self, path, loss="total"):
        self.dir = os.path.abspath(path)
        self.name = os.path.basename(self.dir.rstrip("/\\"))
        opt = self.json("trainsuite.json")
        self.cfg = self.json("config.json")
        self.log_name = opt.get("log", "log.jsonl")
        self.text_name = opt.get("text_log", "train.log")
        self.loss_key = opt.get("loss", loss)
        self.progress = str(opt.get("progress", "step "))      # how a progress line of the text log starts (crash detection)
        self.total = int(opt.get("steps") or self.cfg.get("steps") or 0)
        self.resumes = []                                       # the first step of each later attempt (watch looks at the rows from there on)
        self.train, self.evals, self.restarts = read_log(self.p(self.log_name), self.resumes)

    # ---- files
    def p(self, *a):
        return os.path.join(self.dir, *a)

    def json(self, name):
        try:
            with open(self.p(name), encoding="utf-8") as f:
                d = json.load(f)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def mtime(self, name):
        try:
            return os.path.getmtime(self.p(name))
        except OSError:
            return None

    def heartbeat_age(self):
        """seconds since the newest line of the metrics stream or the text log (None: neither exists)"""
        ts = [t for t in (self.mtime(self.log_name), self.mtime(self.text_name)) if t]
        return time.time() - max(ts) if ts else None

    def text_tail(self, nbytes=20000):
        try:
            with open(self.p(self.text_name), "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - nbytes))
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def crash_reason(self):
        """the last line of a traceback when the text log ends in one (a crashed process writes nothing more), else ''.
        The last traceback of the final 20 KB counts unless a progress line follows it: a line that starts with `self.progress`
        (trainsuite.json "progress", default "step ", the bev_fusion trainer). Not "any line after it": a crash message can run over
        several lines (CUDA errors do), and launchers print after a crash."""
        text = self.text_tail()
        i = text.rfind("Traceback (most recent call last)")
        if i < 0:
            return ""
        after = [x for x in text[i:].splitlines() if x.strip()]
        if self.progress and any(x.startswith(self.progress) for x in after[1:]):
            return ""
        exc = next((x for x in after[1:] if not x.startswith((" ", "	"))), after[-1])   # the exception line, not a launcher's later print
        return exc.strip()[:200]

    def skipped_steps(self):
        """(non-finite steps the trainer skipped, longest run of them in a row) from the text log"""
        try:
            with open(self.p(self.text_name), encoding="utf-8", errors="replace") as f:
                hits = [int(m.group(1)) for m in re.finditer(r"non-finite .*?skipped \((\d+) in a row\)", f.read())]
        except OSError:
            return 0, 0
        return len(hits), max(hits, default=0)

    def stopped(self):
        """the STOPPED marker (dict) when the run ended because of a stop request and has not run since, else None. A trainer with
        the stop check moves STOPPED aside at its start (10-40 s after a restart, while the imports run); until then, and for any other trainer, STOPPED must not
        be older than the last output by more than STOPPED_GRACE_S: a restarted run that has printed for 10 s is not 'stopped' (watch dropped it as terminal)"""
        t = self.mtime("STOPPED")
        if not t:
            return None
        last = max([x for x in (self.mtime(self.log_name), self.mtime(self.text_name)) if x] or [0])
        return (self.json("STOPPED") or {"reason": ""}) if t >= last - STOPPED_GRACE_S else None

    # ---- progress
    @property
    def step(self):
        return int(self.train[-1]["step"]) if self.train else 0

    def state(self):
        if self.mtime(".done"):
            return "done"
        if self.stopped() is not None:
            return "stopped"
        if self.crash_reason():
            return "failed"
        age = self.heartbeat_age()
        if age is None:
            return "new"
        if age > self.stale_limit():
            return "ended" if self.total and self.step >= self.total else "stalled"
        return "running"

    def recent_rows(self, n=30):
        """the last n rows of the CURRENT attempt: back from the end while the steps fall and the time between rows stays below 10 times the
        median interval (a resume leaves the downtime between two attempts in the file). Rows without `t` give the plain last n."""
        r = self.train[-n:]
        if len(r) < 3 or not all("t" in x for x in r):
            return r
        gaps = [b["t"] - a["t"] for a, b in zip(r, r[1:])]
        med = statistics.median(gaps)
        cut = 0
        for i, g in enumerate(gaps):
            if g < 0 or (med > 0 and g > 10 * med):
                cut = i + 1
        return r[cut:]

    def timed(self):
        """True when the rows carry the time `t` (runs from a trainer of 2026-10-31 on)"""
        return bool(self.train) and all("t" in x for x in self.train[-5:])

    def row_interval(self):
        """median seconds between two logged rows of the current attempt (None without row times)"""
        r = self.recent_rows()
        gaps = [b["t"] - a["t"] for a, b in zip(r, r[1:]) if "t" in a and "t" in b and b["t"] > a["t"]]
        return statistics.median(gaps) if gaps else None

    def stale_limit(self):
        """seconds without output before a run counts as stalled: 15 min, longer when a logged row takes longer than 90 s"""
        iv = self.row_interval()
        return max(STALE_S, 10 * iv) if iv else STALE_S

    def s_per_step(self):
        """seconds per step: from the row times `t` over the current attempt when the stream has them, else the trainer's own average
        (an average since its process start, startup and evals included: only a rough speed)"""
        r = self.recent_rows()
        if len(r) >= 2 and all("t" in x for x in r):
            a, b = r[0], r[-1]
            if b["step"] > a["step"] and b["t"] > a["t"]:
                return (b["t"] - a["t"]) / (b["step"] - a["step"])
        return self.train[-1].get("s_per_step") if self.train else None

    def eta_s(self):
        sp = self.s_per_step()
        return sp * max(0, self.total - self.step) if sp and self.total else None

    # ---- values
    def series(self, key):
        rows = self.evals if key.startswith("val.") else self.train
        return [(r["step"], r[key]) for r in rows if key in r]

    def last(self, key, n=10):
        """mean of the last n logged values of key (None: never logged)"""
        v = [x for _, x in self.series(key)][-n:]
        return sum(v) / len(v) if v else None

    def window_mean(self, key, step, n=10):
        """mean of the n logged values of key at or before step (for the check against a reference run)"""
        v = [x for s, x in self.series(key) if s <= step][-n:]
        return sum(v) / len(v) if v else None

    def keys(self):
        ks = []
        for r in self.train:
            for k in r:
                if k not in ks:
                    ks.append(k)
        return ks


def fmt_dur(s):
    if s is None or not math.isfinite(s):
        return "-"
    return f"{s:.0f} s" if s < 90 else f"{s / 60:.0f} min" if s < 5400 else f"{s / 3600:.1f} h"


def median_mad(v):
    m = statistics.median(v)
    return m, statistics.median([abs(x - m) for x in v])
