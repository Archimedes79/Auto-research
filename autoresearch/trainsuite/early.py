"""Early-stop advice from learning curves: does the gap between a run and its control at a fraction of the steps predict the gap at the end?

    python -m trainsuite early <run> --vs <control> --better lower [--key val.total] [--at 0.25] [--band B | --calib calib.json] [--as-rule]
    python -m trainsuite early --calibrate <runs root> --better lower --band B [--key ...] [--out calib.json]
The direction (`--better`, or `--lower KEY`) and the band (seed noise in the units of the key) have no default: a wrong guess calls the better run behind.

`calibrate` takes the FINISHED run pairs of a runs root (same step total, same name prefix before the first `_` unless `--group all`, or the pairs
given with `--pair variant:control`) and, for each key and each fraction f of the steps, compares the gap (variant - control, signed so that
positive means the variant is better) at f with the final gap: how often the signs agree, the Spearman rank correlation, the number of pairs.
With fewer than MIN_PAIRS pairs it says so and gives no threshold. `early` applies the same gap to one running run and says in words what the calibration
supports; 'consider stopping' needs the 95 % lower bound of the sign agreement >= MIN_AGREE from >= MIN_INDEP_PAIRS pairs of >= MIN_INDEP_RUNS distinct runs. It is advice only: exit code 0, nothing is stopped (`watch --enforce` stays the only thing that stops a run). Standard library only.
"""
import itertools
import math
import os

from .kill import Rule
from .runs import Run

FRACTIONS = (0.1, 0.25, 0.5, 0.75)
MIN_PAIRS = 6              # fewer usable pairs: no calibration, no threshold
COUNT_KEYS = ("val.n", "val.n_skipped")   # batches in the validation mean / left out for a non-finite loss (trainer, D73): counts, not metrics
FINAL_FRAC = 0.9           # the final gap is taken at the last evaluation at or after this fraction of the steps
FRAC_TOL = 0.02            # `early` matches a calibrated fraction when the eval fractions used lie within this much of the current one
MIN_AGREE = 0.8            # `early` calls a gap a reason to consider stopping when the 95 % LOWER bound of the sign agreement is at least this
MIN_INDEP_PAIRS = 15       # ... and the pairs beyond the band number at least this many ...
MIN_INDEP_RUNS = 8         # ... and come from at least this many distinct runs (N runs make N(N-1)/2 pairs that share runs)


# ---- statistics
def ranks(v):
    """average ranks (1-based), ties share the mean rank"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def spearman(x, y):
    """rank correlation of two equally long lists; None when fewer than 2 values or one side is constant. Not invariant to flipping single pairs
    (x, y) -> (-x, -y): `oriented` fixes the orientation first"""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


def oriented(ge, gf):
    """each pair (early, final) flipped to a positive final gap: which run of a pair is called the variant must not change the correlation"""
    sg = [-1.0 if y < 0 else 1.0 for y in gf]
    return [x * s for x, s in zip(ge, sg)], [y * s for y, s in zip(gf, sg)]


def direction(key, better, lower=()):
    """'higher' / 'lower' for a key: in the explicit `lower` list, else `better`; never a default (a logged val.* is mostly a loss: a wrong guess calls
    the better run 'behind')"""
    if key in lower:
        return "lower"
    if better in ("higher", "lower"):
        return better
    raise ValueError(f"direction of {key!r} not stated: give --better higher|lower (or --lower {key})")


def wilson_low(k, n, z=1.96):
    """lower end of the 95 % Wilson interval of k successes in n trials (how little a small count proves)"""
    if n == 0:
        return 0.0
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def sign(x):
    return (x > 0) - (x < 0)


# ---- the gap of two runs
def advantage(a, b, sgn):
    """signed difference a - b, positive when a is the better one (sgn = +1 higher is better, -1 lower is better)"""
    return sgn * (a - b)


def gap_at(var, ctl, key, limit, sgn, min_step=0.0):
    """(advantage of var over ctl, step, rows of var up to the step) at the last point at or before `limit` that both runs have, or None.
    A `val.*` key is compared at a common evaluation step; any other key as the mean of the last 10 logged values up to `limit` (both runs must have
    reached it). `min_step` keeps the final gap away from early evaluations."""
    if key.startswith("val."):
        a, b = dict(var.series(key)), dict(ctl.series(key))
        common = sorted(s for s in a if s in b and min_step <= s <= limit + 1e-9)
        if not common:
            return None
        s = common[-1]
        return advantage(a[s], b[s], sgn), s, sum(1 for x in a if x <= s)
    if min(var.step, ctl.step) < limit:
        return None
    x, y = var.window_mean(key, limit), ctl.window_mean(key, limit)
    if x is None or y is None:
        return None
    return advantage(x, y, sgn), limit, sum(1 for s, _ in var.series(key) if s <= limit)


def finished(run):
    return run.total > 0 and run.state() == "done"


def default_keys(runs):
    """the `val.*` keys of the runs (the `*last*` variants of the trainer and the counts `val.n` / `val.n_skipped` of the validation are left out: they are no metric), sorted"""
    ks = {k for r in runs for e in r.evals for k in e if k.startswith("val.") and "last" not in k and k not in COUNT_KEYS}
    return sorted(ks)


# ---- pairs
def find_pairs(root, loss="total", explicit=None, group="prefix"):
    """list of (variant Run, control Run) of finished runs with the same step total. explicit: [(variant, control)] names under root (no further
    checks beyond 'finished and same total'); else every two finished runs of the root with evaluations and, for group 'prefix', the same name
    prefix before the first `_` (t8_ctl_s0 and t8_bevfeat_s1 are a pair; t8_ctl_s0 and t10_dual_s0 are not). The order inside a pair does not matter
    (the sign agreement counts the same, the correlation is taken on pairs oriented by their final gap)."""
    def opn(n):
        return Run(n if os.path.isdir(n) else os.path.join(root, n), loss)
    if explicit:
        cand = [(opn(a), opn(b)) for a, b in explicit]
    else:
        runs = [Run(os.path.join(root, d), loss) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))] if os.path.isdir(root) else []
        runs = [r for r in runs if r.evals]
        key = (lambda r: r.name.split("_")[0]) if group == "prefix" else (lambda r: "")
        cand = [(a, b) for a, b in itertools.combinations(runs, 2) if key(a) == key(b)]
    return [(a, b) for a, b in cand if finished(a) and finished(b) and a.total == b.total]


# ---- calibration
def calibrate(pairs, keys=None, lower=(), better=None, *, band, fractions=FRACTIONS, min_pairs=MIN_PAIRS):
    """{'keys': {key: {'better', 'fractions': {f: entry}}}, ...} for the pairs. The direction of every key must be stated (`better`, or the key in
    `lower`: ValueError otherwise) and so must the `band` (the seed noise in the units of the key). entry: n (pairs with an early and a final gap),
    n_tied (a gap of exactly 0: left out of the sign count), n_sign, frac_lo / frac_hi (the fractions of the evaluations really used), calibrated
    (n_sign >= min_pairs), agree (k, signs equal), n_band / agree_band / n_runs_band (pairs whose early gap exceeds the band, how many of them agree, from how
    many distinct runs), spearman (on pairs oriented by their final gap). Below min_pairs agree and spearman are None."""
    keys = list(keys) if keys else default_keys([r for p in pairs for r in p])
    out = {}
    for key in keys:
        d = direction(key, better, lower)
        sgn = -1.0 if d == "lower" else 1.0
        per_f = {}
        for f in fractions:
            ge, gf, fr, nm = [], [], [], []
            for a, b in pairs:
                fin = gap_at(a, b, key, a.total, sgn, FINAL_FRAC * a.total)
                early = gap_at(a, b, key, f * a.total, sgn)
                if fin is None or early is None or early[1] >= fin[1]:
                    continue
                ge.append(early[0])
                gf.append(fin[0])
                fr.append(early[1] / a.total)
                nm.append((a.name, b.name))
            n_tied = sum(1 for x, y in zip(ge, gf) if x == 0 or y == 0)
            sg = [(x, y, n) for x, y, n in zip(ge, gf, nm) if x != 0 and y != 0]
            ban = [t for t in sg if abs(t[0]) > band]
            ent = dict(n=len(ge), n_tied=n_tied, n_sign=len(sg), frac_lo=min(fr, default=None), frac_hi=max(fr, default=None),
                       calibrated=len(sg) >= min_pairs, agree=None, n_band=len(ban), agree_band=None, n_runs_band=len({r for t in ban for r in t[2]}),
                       spearman=None)
            if ent["calibrated"]:
                ent["agree"] = sum(1 for x, y, _ in sg if sign(x) == sign(y))
                ent["spearman"] = spearman(*oriented(ge, gf))
            if len(ban) >= min_pairs:
                ent["agree_band"] = sum(1 for x, y, _ in ban if sign(x) == sign(y))
            per_f[str(f)] = ent
        out[key] = dict(better=d, fractions=per_f)
    runs = {r.name for p in pairs for r in p}
    return dict(version=1, band=band, min_pairs=min_pairs, n_pairs=len(pairs), n_runs=len(runs), pairs=[[a.name, b.name] for a, b in pairs], keys=out)


def check_calibration(cal):
    """ValueError for a calibration that does not state its direction per key or its band (an old or hand-made file): its counts cannot be read"""
    for key, kd in cal.get("keys", {}).items():
        if kd.get("better") not in ("higher", "lower"):
            raise ValueError(f"the calibration does not state the direction (better) of {key!r}: not used")
    if not isinstance(cal.get("band"), (int, float)):
        raise ValueError("the calibration does not state its band: not used")


def describe(cal):
    """the calibration as text"""
    lines = [f"calibration: {cal['n_pairs']} finished pairs from {cal['n_runs']} runs (pairs share runs: they are not independent), band {cal['band']:g}, "
             f"at least {cal['min_pairs']} usable pairs needed"]
    if cal["n_pairs"] < cal["min_pairs"]:
        lines.append(f"not enough pairs to calibrate ({cal['n_pairs']} < {cal['min_pairs']}): no threshold")
    for key, kd in cal["keys"].items():
        lines.append(f"{key} ({kd['better']} is better)")
        for f, e in kd["fractions"].items():
            where = "no evaluation at or before this fraction in both runs" if not e["n"] else \
                "evaluations at %s of the steps" % (("%.0f %%" % (100 * e["frac_lo"])) if e["frac_lo"] == e["frac_hi"]
                                                      else "%.0f-%.0f %%" % (100 * e["frac_lo"], 100 * e["frac_hi"]))
            if not e["n"]:
                lines.append(f"  f={f}: {where}")
            elif not e["calibrated"]:
                lines.append(f"  f={f}: {where}; not enough pairs to calibrate ({e['n_sign']} with a non-zero early and final gap, {e['n_tied']} tied)")
            else:
                s = f"  f={f}: {where}; sign of the early gap = sign of the final gap in {e['agree']} of {e['n_sign']} pairs " \
                    f"(95 % lower bound {wilson_low(e['agree'], e['n_sign']):.2f}), Spearman " + ("-" if e["spearman"] is None else f"{e['spearman']:.2f}")
                if e["agree_band"] is not None:
                    s += f"; early gap beyond the band: {e['agree_band']} of {e['n_band']} (from {e['n_runs_band']} runs, " \
                         f"lower bound {wilson_low(e['agree_band'], e['n_band']):.2f})"
                else:
                    s += f"; early gap beyond the band: only {e['n_band']} pairs, not calibrated"
                lines.append(s)
    return "\n".join(lines)


# ---- advice for one run
def resolve_band(band, cal):
    """the band `advise` uses: the calibration's own (its counts beyond the band were made with it); a given band that differs is refused; without a
    calibration a band must be given (the default of the lane project, 0.010, is its S band and means nothing for a loss)"""
    if cal is not None:
        if band is not None and abs(band - cal["band"]) > 1e-12:
            raise ValueError(f"--band {band:g} differs from the band of the calibration ({cal['band']:g}): its counts beyond the band were made with that band; "
                             "calibrate again with the band you want")
        return cal["band"]
    if band is None:
        raise ValueError("no band: give --band in the units of the key (the seed noise), or --calib")
    return band


def _entry(cal, key, better, frac):
    """the calibration entry of `key` whose evaluations lie at the current fraction (None: none does, or the calibration used another direction)"""
    kd = (cal or {}).get("keys", {}).get(key)
    if not kd or kd["better"] != better:
        return None
    best = None
    for f, e in kd["fractions"].items():
        if e["n"] and e["frac_lo"] - FRAC_TOL <= frac <= e["frac_hi"] + FRAC_TOL:
            if best is None or e["n_band"] > best["n_band"]:
                best = e
    return best


def advise(var, ctl, key, at, better, band, cal=None, min_rows=3, min_agree=MIN_AGREE):
    """(verdict text, info dict) for a run against its control. at: fraction of the steps to compare at (None: as far as the run has got). `better` is
    required ('higher' / 'lower'). 'consider stopping' needs the 95 % lower bound of the calibrated sign agreement >= min_agree AND at least
    MIN_INDEP_PAIRS pairs from MIN_INDEP_RUNS distinct runs beyond the band; with less it says so."""
    if better not in ("higher", "lower"):
        raise ValueError("direction not stated: better must be 'higher' or 'lower'")
    if cal is not None:
        check_calibration(cal)
    band = resolve_band(band, cal)
    sgn = -1.0 if better == "lower" else 1.0
    if not var.total or var.total != ctl.total:
        return f"cannot compare: step totals differ ({var.total} vs {ctl.total}): another schedule", dict(verdict="none")
    limit = (at * var.total) if at else var.step
    g = gap_at(var, ctl, key, limit, sgn)
    if g is None or g[2] < min_rows:
        n = 0 if g is None else g[2]
        return f"too early: {n} rows of {key} up to step {int(limit)} (need {min_rows}): no gap", dict(verdict="none")
    adv, step, _ = g
    frac = step / var.total
    info = dict(key=key, step=step, frac=frac, advantage=adv, band=band, better=better)
    at_txt = f"{key} at step {int(step)} ({100 * frac:.0f} %)"
    if abs(adv) <= band:
        return f"inside the band: no signal ({at_txt}: {adv:+.4g} against the control, band {band:g})", dict(info, verdict="inside")
    if adv > 0:
        return f"ahead of the control by {adv:.4g} > band {band:g} on {at_txt}: nothing to stop", dict(info, verdict="ahead")
    e = _entry(cal, key, better, frac)
    if e is None:
        return f"not calibrated: gap only ({at_txt}: {adv:+.4g} against the control, behind by more than the band {band:g}; " \
               f"no calibration for this key, direction and fraction)", dict(info, verdict="uncalibrated")
    if e["agree_band"] is None:
        return f"not calibrated: gap only ({at_txt}: {adv:+.4g}; the calibration has only {e['n_band']} pairs with an early gap beyond a band, " \
               f"{cal['min_pairs']} needed)", dict(info, verdict="uncalibrated")
    k, n, nr = e["agree_band"], e["n_band"], e["n_runs_band"]
    lb = wilson_low(k, n)
    base = f"behind control by {-adv:.4g} > band {band:g} on {at_txt} and this key predicted the final sign in {k} of {n} pairs " \
           f"from {nr} runs (95 % lower bound {lb:.2f})"
    note = f" [{key} is a logged curve, not the decision metric of the card]"
    info = dict(info, agree=k, n=n, n_runs=nr, lower_bound=lb)
    if n < MIN_INDEP_PAIRS or nr < MIN_INDEP_RUNS:
        return base + f": not enough independent evidence (need {MIN_INDEP_PAIRS} pairs from {MIN_INDEP_RUNS} runs; pairs share runs)" + note, dict(info, verdict="dependent")
    if lb >= min_agree:
        return base + ": consider stopping" + note, dict(info, verdict="consider_stop")
    return base + f", lower bound below {min_agree:g}: no reliable signal" + note, dict(info, verdict="weak")


def as_rule(ctl, info):
    """a kill.txt line for review, only for the verdict 'consider_stop': the variant's value at the compared step is worse than the control's by more
    than the band. None for every other verdict."""
    if info.get("verdict") != "consider_stop":
        return None
    key, step, better, band = info["key"], int(info["step"]), info["better"], info["band"]
    c = ctl.window_mean(key, step) if not key.startswith("val.") else dict(ctl.series(key)).get(info["step"])
    if c is None:
        return None
    op, thr = ("<", c - band) if better == "higher" else (">", c + band)
    line = f"{key}@{step} {op} {thr:.6g}"
    Rule(line)                                                          # must be a rule the suite reads
    return line
