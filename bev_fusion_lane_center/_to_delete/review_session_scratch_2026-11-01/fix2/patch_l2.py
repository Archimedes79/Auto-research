import sys
ev, t, lg = sys.argv[1:4]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
patch(ev, [
('''    """True if the TEST usage log names test_id as a whole id, the rule of autoresearch/tools/run.py (review bug L2, D24: a substring test let `E` or `2026` pass)"""
    if not test_id:
        return False''', '''    """True if test_id is a run id (E<number>...) and the TEST usage log names it as a whole id, the rule of autoresearch/tools/run.py
    (review bug L2, D24: a substring test let `E` pass; the run-id shape keeps date parts such as `2026` or `10` and words out)"""
    if not test_id or not re.fullmatch(r"E\d+[A-Za-z0-9_.-]*", test_id):
        return False'''),
])
patch(t, [
('''    # same limit as run.py: a free-standing token of the log (a date part such as 2026, a word) still counts as a whole id
    assert EV.test_allowed("2026", LOG) and EV.test_allowed("Mirko", LOG)''', '''    # tokens of the log that are not run ids (date parts, words) never pass: the id must have the shape E<number>...
    assert not EV.test_allowed("2026", LOG) and not EV.test_allowed("10", LOG) and not EV.test_allowed("Mirko", LOG)'''),
])
patch(lg, [
('''        lo, hi, _ = percentile_interval(ds)
        res[f"{layer} {metric}"]''', '''        lo, hi, nv = percentile_interval(ds)
        if nv < n_boot:
            lines.append(f"  note: {layer} {metric}: {n_boot - nv} of {n_boot} resamples had nothing to score; interval over the other {nv}")
        res[f"{layer} {metric}"]'''),
('''            lo, hi, _ = percentile_interval(ds)
            res[f"{cls} AP@{thr:g}"]''', '''            lo, hi, nv = percentile_interval(ds)
            if nv < n_boot:
                lines.append(f"  note: {cls} AP@{thr:g}: {n_boot - nv} of {n_boot} resamples had nothing to score; interval over the other {nv}")
            res[f"{cls} AP@{thr:g}"]'''),
])
print("ok")
