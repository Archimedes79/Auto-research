def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


# N1: the cover radius of the long links follows the tight wiring radius
edit("temp/e320/dev/qends_eval.pysrc", [
    ('''    wp = WIRE_POS if wire_pos is None else float(wire_pos)''', '''    wp = WIRE_POS if wire_pos is None else float(wire_pos)
    cp_ = COVER_POS if wire_pos is None else min(COVER_POS, wp)                                                   # N1 (critic 2026-11-13): a long link is dropped only when a connector's ends are within the WIRING radius: under a tight radius a connector that is not wired to the pieces must not delete the long link of B either'''),
    ('''np.linalg.norm(P[0] - pr[0]) <= COVER_POS and np.linalg.norm(P[-1] - pr[2]) <= COVER_POS for pr in pairs''', '''np.linalg.norm(P[0] - pr[0]) <= cp_ and np.linalg.norm(P[-1] - pr[2]) <= cp_ for pr in pairs'''),
    ('''    assert add_connectors(*g_)[2] == 1 and add_connectors(*g_, wire_pos=1.0)[2] == 0 and add_connectors(*g_, wire_pos=2.5)[2] == 1 and add_connectors(*g_, wire_pos=4.0)[2] == 1''', '''    assert add_connectors(*g_)[2] == 1 and add_connectors(*g_, wire_pos=1.0)[2] == 0 and add_connectors(*g_, wire_pos=2.5)[2] == 1 and add_connectors(*g_, wire_pos=4.0)[2] == 1
    off2_ = [(np.array([2.5, -2.0]), np.array([0.0, 1.0]), np.array([10.5, 2.0]), np.array([0.0, 1.0]))]              # N1: both ends 2.5 m from the delivered long link (piece ends at (0, -2) and (8, 2)): covered, hence dropped, by the default (4 m) and by a 4 m / 2.5 m radius; KEPT under a 1.0 m radius (not wired, so it must not delete the link)
    g2_ = (lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, off2_)
    assert len(add_connectors(*g2_)[0]["pts"]) == 3 and len(add_connectors(*g2_, wire_pos=4.0)[0]["pts"]) == 3 and len(add_connectors(*g2_, wire_pos=2.5)[0]["pts"]) == 3 and len(add_connectors(*g2_, wire_pos=1.0)[0]["pts"]) == 4'''),
])

# N5: parse_variants out of score, with asserts
edit("temp/e320/dev/jmaneuver.pysrc", [
    ('''    snapped = {v for v in names if v.startswith("p") and v.endswith("s") and v[1:-1].replace(".", "", 1).isdigit()}                    # p0.1s = connectors at p >= 0.1 with snapped ends (E311)
    parsed = {v: re.fullmatch(r"p(\\d+(?:\\.\\d+)?)(s|an|a|ar(\\d+(?:\\.\\d+)?))?", v) for v in names}''', '''    snapped, attached, attached_n, attached_r, thr = parse_variants(names)'''),
    ('''    attached_n = {v for v, m in parsed.items() if m and m.group(2) == "an"}
    attached_r = {v: float(m.group(3)) for v, m in parsed.items() if m and m.group(3)}
    attached = {v for v, m in parsed.items() if m and m.group(2) in ("a", "an")} | set(attached_r)
    thr = {v: float(m.group(1)) for v, m in parsed.items() if m}
''', ''''''),
    ('''def score(a):''', '''def parse_variants(names):
    """(snapped, attached, attached_n, attached_r, thr) of the variant names: p<t> connectors of the run's pair logits at probability t | p<t>s snapped ends (E311) | p<t>a attached ends, proximity wiring (E320) | p<t>an attached ends + one nearest piece per end, the curve kept (the E320 primary read-out) |
    p<t>ar<r> attached ends, wiring radius r m; `thr` maps every p-variant to its threshold (plain / oracle / oracle+fit are not in it), snapped / attached / attached_n are sets of names, attached_r maps a name to its radius"""
    parsed = {v: re.fullmatch(r"p(\\d+(?:\\.\\d+)?)(s|an|a|ar(\\d+(?:\\.\\d+)?))?", v) for v in names}
    snapped = {v for v, m in parsed.items() if m and m.group(2) == "s"}
    attached_n = {v for v, m in parsed.items() if m and m.group(2) == "an"}
    attached_r = {v: float(m.group(3)) for v, m in parsed.items() if m and m.group(3)}
    attached = {v for v, m in parsed.items() if m and m.group(2) in ("a", "an")} | set(attached_r)
    thr = {v: float(m.group(1)) for v, m in parsed.items() if m}
    return snapped, attached, attached_n, attached_r, thr


def score(a):'''),
    ('''    print("jmaneuver selftest ok")''', '''    sn, at, an, ar, th = parse_variants(["plain", "p0.5", "p0.1", "p0.1s", "p0.1a", "p0.1an", "p0.1ar1.5", "p0.2ar1.0", "oracle", "oracle+fit", "p0.1x"])        # N5 (critic 2026-11-13): the variant names of the read-outs reach the right settings
    assert th == {"p0.5": 0.5, "p0.1": 0.1, "p0.1s": 0.1, "p0.1a": 0.1, "p0.1an": 0.1, "p0.1ar1.5": 0.1, "p0.2ar1.0": 0.2}, th
    assert sn == {"p0.1s"} and at == {"p0.1a", "p0.1an", "p0.1ar1.5", "p0.2ar1.0"} and an == {"p0.1an"} and ar == {"p0.1ar1.5": 1.5, "p0.2ar1.0": 1.0}, (sn, at, an, ar)
    print("jmaneuver selftest ok")'''),
])

# N4: the attachment statistics of the training log, averaged over the last rows (jdiag attlog)
edit("temp/e320/dev/jdiag.pysrc", [
    ('''    wsl bash tools/env/_py14.sh jdiag.py selftest
''', '''    wsl bash tools/env/_py14.sh jdiag.py attlog e320_a_s0                           the attachment statistics of the training log (att_acc, att_acc_nearest, att_ce, att_bin, att_kp_d, att_d), the mean of the last 10 rows of runs/<run>/log.jsonl weighted by att_n (E320 kill criterion; one row = one batch of 8 frames, a single row is noise)
    wsl bash tools/env/_py14.sh jdiag.py selftest
'''),
    ('''def maps_summary(d):''', '''def attlog(run, last=10, rows=None):
    """E320: the attachment statistics of the training log, the mean over the last `last` rows that carry att_n of runs/<run>/log.jsonl (`rows`: a list of dicts instead, for the test) weighted by att_n; one row is one batch of 8 frames, so a single row is noise (the standard error of att_acc - att_acc_nearest is about 0.05 there). Returns the dict and prints it,
    with att_acc - att_acc_nearest: the kill criterion of the card reads this difference against its margin."""
    if rows is None:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", run, "log.jsonl")) if l.strip()]
    rows = [r for r in rows if r.get("att_n", 0) > 0][-last:]
    assert rows, f"{run}: no row of log.jsonl carries att_n (a run without --attach)"
    w = np.array([r["att_n"] for r in rows], float)
    out = {k: float(np.average([r[k] for r in rows], weights=w)) for k in ("att_acc", "att_acc_nearest", "att_ce", "att_bin", "att_kp_d", "att_d")}
    out.update(rows=len(rows), att_n=float(w.sum()), first_step=rows[0].get("step"), last_step=rows[-1].get("step"), att_acc_minus_nearest=out["att_acc"] - out["att_acc_nearest"])
    print(f"{run}: " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in out.items()))
    return out


def maps_summary(d):'''),
    ('''    elif cmd == "snaprule" and len(sys.argv) > 2:''', '''    elif cmd == "attlog" and len(sys.argv) == 3:
        attlog(sys.argv[2])
    elif cmd == "snaprule" and len(sys.argv) > 2:'''),
    ('''def selftest():
''', '''def selftest():
    t = attlog("x", last=2, rows=[dict(att_n=10, att_acc=0.1, att_acc_nearest=0.1, att_ce=1.0, att_bin=0.0, att_kp_d=1.0, att_d=1.0, step=100), dict(att_n=10, att_acc=0.5, att_acc_nearest=0.4, att_ce=1.0, att_bin=0.0, att_kp_d=1.0, att_d=0.8, step=200),
                                  dict(att_n=30, att_acc=0.8, att_acc_nearest=0.5, att_ce=0.5, att_bin=0.0, att_kp_d=1.0, att_d=0.6, step=300), dict(step=400)])         # E320: the last 2 rows with att_n, weighted by att_n
    assert abs(t["att_acc"] - (0.5 * 10 + 0.8 * 30) / 40) < 1e-9 and abs(t["att_acc_minus_nearest"] - ((0.5 - 0.4) * 10 + (0.8 - 0.5) * 30) / 40) < 1e-9 and t["last_step"] == 300 and t["rows"] == 2, t
'''),
])
print("n edits ok")
