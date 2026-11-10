"""Per-module and timing views of a run: what the trainer's probe writes next to the loss.

Contract keys of a log row (all optional; a trainer that writes none loses nothing):
  grad.<module>   L2 norm of the gradient of one top-level module (`obj_head`, `fusion`, ...), every logged step
  upd.<module>    update-to-weight ratio of that module: |w_after - w_before| / |w_before|, the size of one optimiser step relative to the weights
  t_data, t_step  seconds the step waited for the data loader, seconds of the whole step (data wait included)
  grad_norm       the global gradient norm of the step (`grad_trend`: first rows, now, peak)
and one file per incident in the run directory, `incident_<step>.json`: {"step", "kind" ("nonfinite" | "spike" | ...), "batch": [sample ids], "losses": {...}}.
A module that is frozen on purpose is named in config.json ("freeze": [prefix, ...] or "train_only": [prefix, ...], the prefixes of the model's modules) and is
not asked to learn.
What a run trained on (`data_of`) is read from the run's own files, so a trainer for another dataset plugs in by naming it in its config.json: `dataset` (or `data_root`),
`train_split` / `val_split`, `train_scenes` / `val_scenes` (a list or a count), and `gt.json` with a `*version` and a `*signature` key for the ground truth it was made against.
"""
import json
import math
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


def grad_trend(run, window=WINDOW):
    """the global gradient norm over the run: {first: median of the finite values of the first 10 rows, now: median of the finite values among the last `window` ROWS (None when
    none of them is finite: the gradient is gone), nonfinite_recent: how many of those rows are not finite, base: median of the finite values before that window (the last 100;
    None when there are fewer than `window` of them), ratio: now / base, peak, peak_step, rows}; None when no row has a `grad_norm`"""
    allr = [(r["step"], r["grad_norm"]) for r in run.train if isinstance(r.get("grad_norm"), (int, float)) and not isinstance(r.get("grad_norm"), bool)]
    if not allr:
        return None
    fin = lambda rows: [v for _, v in rows if math.isfinite(v)]            # noqa: E731
    recent, first, before = fin(allr[-window:]), fin(allr[:10]), fin(allr[:-window])[-100:]
    now = statistics.median(recent) if recent else None
    base = statistics.median(before) if len(before) >= window else None
    pairs = [(s, v) for s, v in allr if math.isfinite(v)]
    peak_step, peak = max(pairs, key=lambda p: p[1]) if pairs else (None, None)
    return dict(first=statistics.median(first) if first else None, now=now, nonfinite_recent=len(allr[-window:]) - len(recent), base=base,
                ratio=(now / base if (now is not None and base) else None), peak=peak, peak_step=None if peak_step is None else int(peak_step), rows=len(allr))


def _count(v):
    """a scene list or a count -> the number (0 and anything that is no number: None, a list that is empty says nothing)"""
    n = len(v) if isinstance(v, (list, tuple)) else (int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != math.inf else 0)
    return n or None


def _scalar(v):
    """a version or signature: a string or a number (no bool), as text; else ''"""
    return str(v) if isinstance(v, (str, int, float)) and not isinstance(v, bool) and v == v else ""


def data_key(run):
    """what the run says about its data, as a dict: dataset, train / val {label, n, ids} (ids: a short hash of the scene list when it is a list), gt_version, gt_signature. Every field
    is '' / None when the run does not name it (config.json and gt.json, see the head of this file; a trainer of another dataset plugs in by naming itself there)"""
    import hashlib
    c = run.cfg if isinstance(run.cfg, dict) else {}
    name = c.get("dataset") if isinstance(c.get("dataset"), str) else ""
    if not name and isinstance(c.get("data_root"), str):
        name = os.path.basename(c["data_root"].rstrip("/\\"))
    out = {"dataset": name}
    for part in ("train", "val"):
        label, scenes = c.get(f"{part}_split"), c.get(f"{part}_scenes")
        label = label if isinstance(label, str) else ""
        n = _count(scenes)
        ids = hashlib.sha1("\n".join(sorted(str(x) for x in scenes)).encode()).hexdigest()[:8] if isinstance(scenes, (list, tuple)) and scenes else ""
        out[part] = {"label": label, "n": n, "ids": ids}
    gt = run.json("gt.json")
    out["gt_version"] = next((_scalar(v) for k, v in gt.items() if k.endswith("version") and _scalar(v)), "")
    out["gt_signature"] = next((_scalar(v) for k, v in gt.items() if k.endswith("signature") and _scalar(v)), "")
    return out


def data_of(run):
    """what the run trained on, in one short line (`<dataset> <train split> <n> / <val split> <n> scenes, gt <version> <signature>`, whatever of it the run records); '' when it
    names nothing. Nothing here knows a dataset"""
    k = data_key(run)
    items = [" ".join(x for x in (k[p]["label"] or p, "" if k[p]["n"] is None else str(k[p]["n"])) if x) for p in ("train", "val") if k[p]["label"] or k[p]["n"]]
    head = " ".join(x for x in (k["dataset"], " / ".join(items) + (" scenes" if any(k[p]["n"] for p in ("train", "val")) else "")) if x.strip())
    tail = " ".join(x for x in ("gt", k["gt_version"], k["gt_signature"]) if x) if (k["gt_version"] or k["gt_signature"]) else ""
    return ", ".join(x for x in (head, tail) if x)


def data_differs(run, ref):
    """'' when the two runs trained on the same data as far as BOTH say so, else what differs. A field that only one of them names is no difference (a run from before gt.json
    existed knows nothing); the ground truth is compared by its signature when both have one (a version label that differs over the same signature is only a name)"""
    a, b, diff = data_key(run), data_key(ref), []
    if a["dataset"] and b["dataset"] and a["dataset"] != b["dataset"]:
        diff.append(f"dataset {a['dataset']} against {b['dataset']}")
    for part in ("train", "val"):
        x, y = a[part], b[part]
        if x["label"] and y["label"] and x["label"] != y["label"]:
            diff.append(f"{part} split {x['label']} against {y['label']}")
        if x["n"] and y["n"] and x["n"] != y["n"]:
            diff.append(f"{part} scenes {x['n']} against {y['n']}")
        elif x["ids"] and y["ids"] and x["ids"] != y["ids"]:
            diff.append(f"other {part} scenes ({x['n']} each)")
    if a["gt_signature"] and b["gt_signature"]:
        if a["gt_signature"] != b["gt_signature"]:
            diff.append(f"ground truth {a['gt_signature']} against {b['gt_signature']}")
    elif a["gt_version"] and b["gt_version"] and a["gt_version"] != b["gt_version"]:
        diff.append(f"ground truth version {a['gt_version']} against {b['gt_version']}")
    return "; ".join(diff)


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
