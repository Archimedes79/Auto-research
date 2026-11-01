"""Health checks of a run: does the training look sound? (It says nothing about the QUALITY of the network: that is the evaluation.)

Every check returns a Finding(level, code, text). Levels: alert (act now: crashed, stalled, non-finite), warn (look at it), info.
Thresholds are module constants; the loss is noisy per batch, so every test compares means of windows and uses the spread of the window.
`check(run, ref, since_step)`: spike and grad-norm tests look at every row after `since_step` (watch passes the step of its last look, so a
spike between two looks is not missed); without it they look at the last 5 rows.
"""
import math
import statistics
from collections import namedtuple

from . import modules
from .runs import fmt_dur, median_mad

Finding = namedtuple("Finding", "level code text")
RANK = {"alert": 0, "warn": 1, "info": 2}
SPIKE_K = 8.0            # a single value this many robust sigmas above the recent median
RISE_K = 3.0             # window mean rise in standard errors
SHIFT_K, SHIFT_REL = 5.0, 0.30     # a level shift: last 10 rows against the 30 before, in standard errors and relative
SLOW_FACTOR = 1.5        # recent s/step against the typical one
GPU_LIMIT_GB = 11.0      # project rule: GPU < 11 GB
REF_TOL, REF_K, REF_ROWS = 0.10, 3.0, 50     # against a reference: relative gap, standard errors, rows per window
MIN_ROWS = 40
DEAD_GRAD = 1e-10        # a module whose gradient norm (median of the last 20 rows) is not above this learns nothing (detached, dead activations, frozen by mistake)
GRAD_MODULE_K = 10.0     # a module's gradient norm this many times its own median before the window
UPD_LOW, UPD_HIGH = 1e-7, 0.2     # update-to-weight ratio per step: below, the module does not move; above, one step changes its weights by 20 % (Adam moves each weight by about lr,
                         # 3e-4 .. 6e-4, and weights are 0.01 .. 0.05: 1e-2 is normal; not calibrated on a real probe run yet, see README)
DATA_BOUND = 0.30        # the part of a step that waits for the data loader
RAMP_FRAC = 0.5          # before this share of the steps a rising loss is only info: the bev_fusion `total` rises from 8.5 to 12.5 between
                         # steps 500 and 1300 in every run (cause not analysed), also in the references; the reliable early check is the
                         # comparison with a reference run (`vs_ref`)


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


def _shift(vals):
    """(rise of the last 10 values over the 30 before, standard error) or None: catches a jump that stays, which `_rise` sees only
    while the two windows straddle it"""
    if len(vals) < 40:
        return None
    a, b = vals[-10:], vals[-40:-10]
    se = math.sqrt(statistics.pvariance(a) / 10 + statistics.pvariance(b) / 30) or 1e-12
    return sum(a) / 10 - sum(b) / 30, se


def _scan(rows, key, since_step):
    """(baseline values, values to test) of a key: the rows after since_step (at most 50; default the last 5) against those before"""
    pairs = [(r["step"], r[key]) for r in rows if key in r and math.isfinite(r[key])]
    n_scan = sum(1 for s, _ in pairs if since_step is not None and s > since_step) if since_step is not None else 5
    n_scan = min(max(n_scan, 1), 50)
    return [v for _, v in pairs[:-n_scan]][-100:], [v for _, v in pairs[-n_scan:]]


def check(run, ref=None, since_step=None):
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
        base, scan = _scan(rows, key, since_step)
        if len(base) >= MIN_ROWS - 5:
            m, mad = median_mad(base)
            sigma = max(1.4826 * mad, 0.05 * abs(m), 1e-9)
            if max(scan) - m > SPIKE_K * sigma:
                add("warn", "spike", f"{key} = {max(scan):.3g} against a recent median {m:.3g} (> {SPIKE_K:g} sigma)")
        r = _rise(vals)
        if r and (frac is None or frac >= 0.2):
            d, se, w = r
            if d > RISE_K * se:
                add(RISE_LEVEL(frac), "rising", f"{key} rose by {d:.3g} over the last {w} rows (> {RISE_K:g} standard errors)"
                    + ("" if frac is None or frac >= RAMP_FRAC else "; the multi-task loss ramps early on, compare with a reference"))
            elif frac is not None and frac < 0.8 and abs(d) < se:
                add("info", "flat", f"{key} did not move over the last {w} rows (change {d:+.3g}, standard error {se:.2g})")
        s = _shift(vals)
        if s and (frac is None or frac >= 0.2) and s[0] > SHIFT_K * s[1] and s[0] > SHIFT_REL * abs(sum(vals[-40:-10]) / 30):
            add(RISE_LEVEL(frac), "shift", f"{key} jumped by {s[0]:.3g} ({s[0] / abs(sum(vals[-40:-10]) / 30):+.0%}) against the 30 rows before and stays")
        for k in (k for k in run.keys() if k.startswith("group.")):
            g = [v for _, v in run.series(k) if math.isfinite(v)]
            r = _rise(g) if len(g) >= MIN_ROWS else None
            if r and (frac is None or frac >= 0.2) and r[0] > RISE_K * r[1]:
                add(RISE_LEVEL(frac), "rising", f"{k} rose by {r[0]:.3g} over the last {r[2]} rows")
    gbase, gscan = _scan(rows, "grad_norm", since_step)
    gbase = gbase[max(1, len(gbase) // 20):]
    if len(gbase) >= MIN_ROWS - 5:
        m, _ = median_mad(gbase)
        if max(gscan) > 10 * m:
            add("warn", "grad_norm", f"grad_norm {max(gscan):.3g} is more than 10 times its median {m:.3g}")
    r = [x for x in run.recent_rows(200) if "t" in x]
    if run.timed() and st in ("running", "stalled") and len(r) >= 12:
        iv = [(b["t"] - a["t"]) / (b["step"] - a["step"]) for a, b in zip(r, r[1:]) if b["step"] > a["step"] and b["t"] > a["t"]]
        if len(iv) >= 12 and statistics.median(iv[-5:]) > SLOW_FACTOR * statistics.median(iv):
            add("warn", "slow", f"recent {statistics.median(iv[-5:]):.2f} s/step against a typical {statistics.median(iv):.2f} (shared box?)")
    gpu = [r["gpu_gb"] for r in rows if "gpu_gb" in r]
    if gpu and gpu[-1] > GPU_LIMIT_GB:
        add("warn", "gpu", f"the trainer's peak GPU allocation {gpu[-1]:.1f} GB is above the {GPU_LIMIT_GB:g} GB limit")
    for m in modules.module_table(run):
        if m["frozen"] or m["rows"] < 10:
            continue
        if m["grad"] is not None and m["grad"] <= DEAD_GRAD:
            add("warn", "dead_module", f"{m['module']}: gradient norm {m['grad']:.2g} over the last {m['rows']} rows (detached, frozen by mistake, or unused on purpose: a switched-off branch, a loss weight of 0?)")
        elif m["grad"] is not None and m["grad_base"] and m["grad"] > GRAD_MODULE_K * m["grad_base"]:
            add("warn", "grad_module", f"{m['module']}: gradient norm {m['grad']:.3g} is {m['grad'] / m['grad_base']:.0f} times its earlier median {m['grad_base']:.3g}")
        if m["upd"] is not None and not (UPD_LOW <= m["upd"] <= UPD_HIGH) and m["grad"] and m["grad"] > DEAD_GRAD:
            add("warn", "update_ratio", f"{m['module']}: one step changes its weights by {m['upd']:.2g} of their size (expected {UPD_LOW:g} .. {UPD_HIGH:g})")
    tm = modules.timing(run)
    if tm and tm["share"] > DATA_BOUND:
        add("warn", "data_bound", f"{100 * tm['share']:.0f} % of a step ({tm['t_data']:.2f} of {tm['t_step']:.2f} s) waits for the data loader")
    inc = modules.incidents(run)
    if inc:
        add("warn", "incident", f"{len(inc)} incident dump(s), the last at step {inc[-1]['step']}: {inc[-1]['kind']} ({inc[-1]['file']})")
    ev = [(s, v) for s, v in run.series("val.total") if math.isfinite(v)]
    if len(ev) >= 2 and ev[-1][1] > 1.05 * ev[-2][1] and len(vals) >= MIN_ROWS:
        a, b = run.window_mean(key, ev[-2][0]), run.window_mean(key, ev[-1][0])
        if a is not None and b is not None and b < a:
            add("warn", "overfit", f"val.total rose {ev[-2][1]:.3g} -> {ev[-1][1]:.3g} while {key} fell {a:.3g} -> {b:.3g}")
    if ref is not None and ref.train and run.step > 0 and run.total and ref.total and run.total != ref.total:
        add("info", "vs_ref", f"{ref.name} has {ref.total} steps and this run {run.total}: at the same step the two are on different schedules (warm-up, learning rate), so "
                              "their losses are not compared")
    elif ref is not None and ref.train and run.step > 0:
        step = min(run.step, ref.step)
        va = [v for s, v in run.series(key) if s <= step and math.isfinite(v)][-REF_ROWS:]
        vb = [v for s, v in ref.series(key) if s <= step and math.isfinite(v)][-REF_ROWS:]
        if len(va) >= 5 and len(vb) >= 5:
            a, b = sum(va) / len(va), sum(vb) / len(vb)
            se = math.sqrt(statistics.pvariance(va) / len(va) + statistics.pvariance(vb) / len(vb)) or 1e-12
            rel = (a - b) / abs(b) if b else 0.0
            worse = rel > REF_TOL and (a - b) > REF_K * se and (frac is None or frac >= 0.2)
            add("warn" if worse else "info", "vs_ref",
                f"{key} {a:.4g} against {ref.name} {b:.4g} at step {step} ({rel:+.1%}, standard error {se:.2g}, last {min(len(va), len(vb))} rows)")
    out.sort(key=lambda f: RANK[f.level])
    return out
