"""Health checks of a run: does the training look sound? (It says nothing about the QUALITY of the network: that is the evaluation.)

Every check returns a Finding(level, code, text). Levels: alert (act now: crashed, stalled, non-finite), warn (look at it), info.
Thresholds are module constants; the loss is noisy per batch, so every test compares means of windows and uses the spread of the window.
"""
import math
import statistics
from collections import namedtuple

from .runs import fmt_dur, median_mad

Finding = namedtuple("Finding", "level code text")
RANK = {"alert": 0, "warn": 1, "info": 2}
SPIKE_K = 8.0            # a single value this many robust sigmas above the recent median
RISE_K = 3.0             # window mean rise in standard errors
SLOW_FACTOR = 1.5        # recent s/step against the typical one
GPU_LIMIT_GB = 11.0      # project rule: GPU < 11 GB
REF_TOL = 0.10           # relative gap to the reference run that is worth a warning
MIN_ROWS = 40
RAMP_FRAC = 0.5          # before this share of the steps a rising loss is only info: the bev_fusion `total` rises from 8.5 to 12.5 between
                         # steps 500 and 1300 in every run (loss ramps, learned task weights), also in the references; the reliable
                         # early check is the comparison with a reference run (`vs_ref`)


def RISE_LEVEL(frac):
    return "warn" if frac is None or frac >= RAMP_FRAC else "info"


def _rise(vals):
    """(mean of the last window - mean of the window before, standard error of that difference, window) or None"""
    w = max(10, len(vals) // 8)
    if len(vals) < 2 * w:
        return None
    a, b = vals[-w:], vals[-2 * w:-w]
    se = math.sqrt(statistics.pvariance(a) / w + statistics.pvariance(b) / w) or 1e-12
    return sum(a) / w - sum(b) / w, se, w


def check(run, ref=None):
    out = []
    add = lambda lvl, code, text: out.append(Finding(lvl, code, text))
    st = run.state()
    age = run.heartbeat_age()
    if st == "failed":
        add("alert", "failed", f"crashed: {run.crash_reason()}")
    elif st == "stalled":
        add("alert", "stalled", f"no new output for {fmt_dur(age)}")
    rows = run.train
    n_skip, in_row = run.skipped_steps()
    if n_skip:
        add("alert" if in_row >= 3 else "warn", "skipped", f"{n_skip} non-finite step(s) skipped by the trainer (up to {in_row} in a row)")
    if not rows:
        return out or [Finding("info", "no_metrics", "no metrics logged yet")]
    bad = [int(r["step"]) for r in rows if any(not math.isfinite(v) for v in r.values())]
    if bad:
        add("alert", "nonfinite", f"non-finite values in the logged rows (first at step {bad[0]}, {len(bad)} rows)")
    key = run.loss_key
    vals = [v for _, v in run.series(key) if math.isfinite(v)]
    frac = run.step / run.total if run.total else None
    if len(vals) >= MIN_ROWS:
        m, mad = median_mad(vals[-100:])
        sigma = max(1.4826 * mad, 0.05 * abs(m), 1e-9)
        if vals[-1] - m > SPIKE_K * sigma:
            add("warn", "spike", f"{key} = {vals[-1]:.3g} against a recent median {m:.3g} (> {SPIKE_K:g} sigma)")
        r = _rise(vals)
        if r and (frac is None or frac >= 0.2):
            d, se, w = r
            if d > RISE_K * se:
                add(RISE_LEVEL(frac), "rising", f"{key} rose by {d:.3g} over the last {w} rows (> {RISE_K:g} standard errors)"
                    + ("" if frac is None or frac >= RAMP_FRAC else "; loss ramps and learned task weights make this normal early on, compare with a reference"))
            elif frac is not None and frac < 0.8 and abs(d) < se:
                add("info", "flat", f"{key} did not move over the last {w} rows (change {d:+.3g}, standard error {se:.2g})")
        for k in (k for k in run.keys() if k.startswith("group.")):
            g = [v for _, v in run.series(k) if math.isfinite(v)]
            r = _rise(g) if len(g) >= MIN_ROWS else None
            if r and (frac is None or frac >= 0.2) and r[0] > RISE_K * r[1]:
                add(RISE_LEVEL(frac), "rising", f"{k} rose by {r[0]:.3g} over the last {r[2]} rows")
    gn = [r["grad_norm"] for r in rows if "grad_norm" in r and math.isfinite(r["grad_norm"])]
    gn = gn[max(1, len(gn) // 20):]
    if len(gn) >= MIN_ROWS:
        m, _ = median_mad(gn[:-1])
        if gn[-1] > 10 * m:
            add("warn", "grad_norm", f"grad_norm {gn[-1]:.3g} is more than 10 times its median {m:.3g}")
    if all("t" in r for r in rows[-30:]) and len(rows) >= 12:
        iv = [(b["t"] - a["t"]) / (b["step"] - a["step"]) for a, b in zip(rows, rows[1:]) if b["step"] > a["step"] and b["t"] > a["t"]]
        if len(iv) >= 12 and statistics.median(iv[-5:]) > SLOW_FACTOR * statistics.median(iv):
            add("warn", "slow", f"recent {statistics.median(iv[-5:]):.2f} s/step against a typical {statistics.median(iv):.2f} (shared box?)")
    else:
        sp = [r["s_per_step"] for r in rows if "s_per_step" in r]
        if len(sp) >= 12 and sp[-1] > SLOW_FACTOR * min(sp[3:]):
            add("warn", "slow", f"average {sp[-1]:.2f} s/step against {min(sp[3:]):.2f} earlier (shared box?)")
    gpu = [r["gpu_gb"] for r in rows if "gpu_gb" in r]
    if gpu and gpu[-1] > GPU_LIMIT_GB:
        add("warn", "gpu", f"GPU memory {gpu[-1]:.1f} GB is above the {GPU_LIMIT_GB:g} GB limit")
    ev = [(s, v) for s, v in run.series("val.total") if math.isfinite(v)]
    if len(ev) >= 2 and ev[-1][1] > 1.05 * ev[-2][1] and len(vals) >= MIN_ROWS:
        a, b = run.window_mean(key, ev[-2][0]), run.window_mean(key, ev[-1][0])
        if a is not None and b is not None and b < a:
            add("warn", "overfit", f"val.total rose {ev[-2][1]:.3g} -> {ev[-1][1]:.3g} while {key} fell {a:.3g} -> {b:.3g}")
    if ref is not None and ref.train and run.step > 0:
        step = min(run.step, ref.step)
        a, b = run.window_mean(key, step), ref.window_mean(key, step)
        if a is not None and b:
            rel = (a - b) / abs(b)
            lvl = "warn" if rel > REF_TOL and (frac is None or frac >= 0.2) else "info"
            add(lvl, "vs_ref", f"{key} {a:.4g} against {ref.name} {b:.4g} at step {step} ({rel:+.1%})")
    out.sort(key=lambda f: RANK[f.level])
    return out
