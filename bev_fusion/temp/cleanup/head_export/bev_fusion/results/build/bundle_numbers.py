"""The numbers of the bundle chapter, in one place: the table of the new run against its control (stored evaluation tables, the idle-GPU timing, the paired bootstrap of
`bev_suite.py compare`), the reading rule, the weak-spot rule, and the compact key table of the summary. The rows (name, key, direction, weak or not) come from the
`bundle.rows` of report_runs.json; nothing is typed here. A value that a run lacks stops the build, except where the row says `new_only` (a head the control never had).

Reading rule (paired bootstrap, b = new, a = control): the interval excludes 0 -> `better` / `worse` by the direction of the row, else `not separated`; no paired row -> `n/a`,
and the paired cell says why (no paired test for the metric / the control has no value); the difference column always keeps the unpaired difference
(`compare` bootstraps the headline numbers of the tasks only (evaluation/bev_task.py BevSegMetric.headline): rows outside them have no paired value; a whole missing group, e.g. all `vel.`, stops the build: a flag was left out).
Weak-spot rule: a row with `weak: true` is listed, lowest value of the new run first, and flagged `weak` when its value is below `bundle.weak_below`."""
import re

from report_lib import compare_key, esc, fail, table_html


def paired_ids(head):
    """('t16_full_s0', 't18_fix') from the header line of a compare output ('a = runs/t16_full_s0, b = runs/t18_fix')"""
    m = re.search(r"a = (\S+), b = (\S+?);", head)
    return tuple(x.replace("\\", "/").rstrip("/").split("/")[-1] for x in m.groups()) if m else None


def reading(row_lower, paired):
    if paired is None:
        return "n/a"
    if not paired[5]:
        return "not separated"
    return "better" if (float(paired[2]) < 0) == row_lower else "worse"


def collect(specs, new, ctl, paired, head, test=False):
    """one dict per row: name, key, lower, weak, new, ctl, diff, paired (the strings of the compare file or None), reading.
    new / ctl: {'id', 'table', 'ms' (None without a timing), 'idle'}; paired / head: read_paired of the compare file"""
    ids, groups = paired_ids(head), {k.split(".")[0] for k in paired}
    if ids != (ctl["id"], new["id"]):
        fail(f"the compare file is about {ids}, the chapter wants a = {ctl['id']} (control), b = {new['id']} (new run)")
    rows = []
    for s in specs:
        lower = bool(s.get("lower"))
        if s.get("timing"):
            for side in (new, ctl):
                if side["ms"] is None or not side["idle"]:
                    fail(f"row '{s['name']}': run {side['id']} has no timing_modules.json, or it was not taken on an idle GPU (not quotable)")
            vals, pair = [new["ms"], ctl["ms"]], None
        else:
            vals = [side["table"].get(s["key"]) for side in (new, ctl)]
            for side, v, role in ((new, vals[0], "new"), (ctl, vals[1], "control")):
                if v is None and not test and not (s.get("new_only") and role == "control"):
                    fail(f"row '{s['name']}': run {side['id']} has no key {s['key']} in its evaluation table")
            ck = compare_key(s["key"])
            pair = paired.get(ck)
            if pair is None and not test and ck.split(".")[0] not in groups:        # the bootstrap covers the headline numbers only, but a whole group missing means a flag was left out
                fail(f"row '{s['name']}': the compare file has no row of the group '{ck.split('.')[0]}' (was it made with --curves --fusion --velocity?)")
            if pair is not None:
                for v, p, role in ((vals[0], pair[1], "new run"), (vals[1], pair[0], "control")):
                    if v is None or abs(v - float(p)) > 1e-3:
                        fail(f"row '{s['name']}': the compare file says {role} {p}, the evaluation table says {v}")
        diff = None if vals[0] is None or vals[1] is None else vals[0] - vals[1]
        rows.append(dict(name=s["name"], key=s.get("key"), lower=lower, weak=bool(s.get("weak")), timing=bool(s.get("timing")), new=vals[0], ctl=vals[1], diff=diff,
                         paired=pair, reading=reading(lower, pair)))
    return rows


def label(row):
    return esc(row["name"] + (", lower is better" if row["lower"] else ""))


def num(v, row):
    return "-" if v is None else (f"{v:.1f}" if row["timing"] else f"{v:.3f}")


def diff_cell(row, band):
    d = row["diff"]
    if d is None:
        return "-"
    if row["timing"]:
        return f"{d:+.1f}"
    cls = "" if abs(d) < band else (" good" if (d < 0) == row["lower"] else " bad")
    return f'<span class="delta{cls}">{d:+.3f}</span>'


def paired_cell(row):
    p = row["paired"]
    if p is not None:
        return f"{p[2]} [{p[3]}, {p[4]}]"
    return "n/a (the control has no such value)" if row["new"] is None or row["ctl"] is None else "n/a (no paired test for this metric)"      # the difference column keeps the unpaired difference


def table_full(rows, new_label, ctl_label, band):
    """metric | new | control | difference | paired b - a [95 % interval] | reading"""
    body = [[label(r), num(r["new"], r), num(r["ctl"], r), diff_cell(r, band), paired_cell(r), r["reading"]] for r in rows]
    return table_html(["metric", new_label, ctl_label, "difference", "paired b - a [95 % interval]", "reading"], body, left=(0, 5))


def table_key(rows, new_label, ctl_label):
    """the compact table of the summary: metric | new | control | reading"""
    body = [[label(r), num(r["new"], r), num(r["ctl"], r), r["reading"]] for r in rows]
    return table_html(["metric", new_label, ctl_label, "against the control (paired bootstrap)"], body, left=(0, 3))


def table_weak(rows, gaps, threshold, new_label, ctl_label):
    """the outputs with `weak: true`, lowest of the new run first: new | control | train minus validation (new run) | paired difference | flag"""
    pick = sorted((r for r in rows if r["weak"] and r["new"] is not None), key=lambda r: r["new"])
    body = [[label(r), num(r["new"], r), num(r["ctl"], r), "-" if gaps.get(r["key"]) is None else f"{gaps[r['key']]:+.3f}", paired_cell(r),
             "weak" if r["new"] < threshold else ""] for r in pick]
    return table_html(["output", new_label, ctl_label, "train - validation gap (new run)", "paired b - a [95 % interval]", f"below {threshold:g}"], body, left=(0, 5))
