"""Per-module and timing views of a run: what the trainer's probe writes next to the loss.

Contract keys of a log row (all optional; a trainer that writes none loses nothing):
  grad.<module>   L2 norm of the gradient of one top-level module (`obj_head`, `fusion`, ...), every logged step
  upd.<module>    update-to-weight ratio of that module: |w_after - w_before| / |w_before|, the size of one optimiser step relative to the weights
  t_data, t_step  seconds the step waited for the data loader, seconds of the whole step (data wait included)
and one file per incident in the run directory, `incident_<step>.json`: {"step", "kind" ("nonfinite" | "spike" | ...), "batch": [sample ids], "losses": {...}}.
A module that is frozen on purpose is named in config.json ("freeze": [prefix, ...] or "train_only": [prefix, ...], the prefixes of the model's modules) and is
not asked to learn.
"""
import json
import os
import re
import statistics

WINDOW = 20          # rows of the recent window of a module / timing view


def _prefixes(value):
    """the module prefixes of a config value: a list / tuple of strings, or one string; anything else (true, 1, null) names nothing. 'fusion', 'fusion.' and 'fusion.*' are one spelling"""
    items = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else []
    return [p.strip().rstrip("*").rstrip(".") for p in items if isinstance(p, str) and p.strip().rstrip("*").rstrip(".")]


def _rel(a, b):
    """do two module prefixes name the same module or one inside the other? ('fusion', 'fusion.') yes; ('lane.res', 'lane') yes; ('lane', 'lanehead') no"""
    a, b = a.rstrip(".") + ".", b.rstrip(".") + "."
    return a.startswith(b) or b.startswith(a)


def _covers(prefix, module):
    """does the prefix name this module or a module that holds it? ('head', 'head') and ('head', 'head.a') yes; a prefix INSIDE the module ('head.gate' for 'head') does not: the rest of it trains"""
    return (module.rstrip(".") + ".").startswith(prefix.rstrip(".") + ".")


def is_frozen(run, module):
    """is the module frozen on purpose by the run's config (freeze: a prefix covers it; train_only: no prefix touches it)"""
    cfg = run.cfg if isinstance(run.cfg, dict) else {}
    freeze, only = _prefixes(cfg.get("freeze")), _prefixes(cfg.get("train_only"))
    if any(_covers(p, module) for p in freeze):
        return True
    return bool(only) and not any(_rel(module, p) for p in only)


def module_names(run):
    names = []
    for r in run.train[-WINDOW:]:
        for k in r:
            if k.startswith(("grad.", "upd.")) and k.split(".", 1)[1] not in names:
                names.append(k.split(".", 1)[1])
    return sorted(names)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def _vals(rows, key):
    return [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]


def module_table(run, window=WINDOW):
    """[{module, rows, grad, grad_base, upd, frozen}] sorted by name: medians of the last `window` rows of the current attempt (grad_base: median of the rows
    before the window, None when there are fewer than `window` of them)"""
    rows = run.train[-max(window, 200):]                      # not recent_rows: that cuts at every pause (an evaluation, a resume) and can leave a handful of rows
    recent, base = rows[-window:], rows[:-window]
    out = []
    for m in module_names(run):
        g, u, gb = _vals(recent, "grad." + m), _vals(recent, "upd." + m), _vals(base, "grad." + m)
        out.append(dict(module=m, rows=len(g), grad=statistics.median(g) if g else None, grad_base=statistics.median(gb) if len(gb) >= window else None,
                        upd=statistics.median(u) if u else None, frozen=is_frozen(run, m)))
    return out


def timing(run, window=50):
    """{t_data, t_step, share} medians over the last `window` rows that carry both keys (share = the part of a step that waited for the data), else None"""
    rows = [r for r in run.train[-max(window, 200):] if _num(r.get("t_data")) and _num(r.get("t_step")) and r["t_step"] > 0][-window:]
    if len(rows) < 5:
        return None
    td, ts = statistics.median(r["t_data"] for r in rows), statistics.median(r["t_step"] for r in rows)
    return dict(t_data=td, t_step=ts, share=td / ts, rows=len(rows))


def incidents(run):
    """[{step, kind, file, ...}] of the incident_<step>.json files, by step; an unreadable file is an incident of kind 'unreadable' (the dump itself failed)"""
    out = []
    try:
        names = [n for n in os.listdir(run.dir) if n.startswith("incident_") and n.endswith(".json")]      # listdir, not glob: a run folder name with [ is a pattern to glob
    except OSError:
        names = []
    for f in (os.path.join(run.dir, n) for n in names):
        m = re.search(r"incident_(\d+)\.json$", f)
        step = int(m.group(1)) if m else -1
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            d = d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            d = {"kind": "unreadable"}
        d.setdefault("step", step)
        d["file"] = os.path.basename(f)
        d.setdefault("kind", "?")
        out.append(d)
    return sorted(out, key=lambda d: d["step"] if isinstance(d["step"], (int, float)) else -1)
