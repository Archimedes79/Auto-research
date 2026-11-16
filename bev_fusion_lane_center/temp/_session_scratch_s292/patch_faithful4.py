import io
def edit(p, pairs):
    s = io.open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, a[:90], s.count(a))
        s = s.replace(a, b)
    io.open(p, "w", encoding="utf-8").write(s)

# ---- jwire_data.py: ds2.missing assert, restore both flags (try / finally), reversed-order join test, flags restored
edit("evaluation/jwire_data.py", [
 ('''    LD.QData.DEFAULT_ENDS = old_ends
    LD.QData.DEFAULT_JUNCTION = True
    ds2, dl2 = LD.loader("val_clean", bs=DUMP_PAIR["bs"], workers=DUMP_PAIR["workers"], shuffle=False, map_mode=DUMP_PAIR["map_mode"], pixels=False)
    by_row = {}
    for b in dl2:                                                                                          # pass 2: the pair model on its dump's loader; a frame is kept when B's pass saw it
        with torch.no_grad():
            _, _, o = mp(b["x"].to(dev).float(), gt=None)
        for i in range(len(b["row"])):
            if int(b["row"][i]) in b_side and int(b["row"][i]) not in by_row:
                by_row[int(b["row"][i])] = frame_dict(o, i, ends)
        if all(r in by_row for r in rows):
            break
''', '''    LD.QData.DEFAULT_ENDS = old_ends
    old_junction = getattr(LD.QData, "DEFAULT_JUNCTION", False)
    LD.QData.DEFAULT_JUNCTION = True                                                                       # as `jends_eval.dump` sets it before its loader; put back below
    try:
        ds2, dl2 = LD.loader("val_clean", bs=DUMP_PAIR["bs"], workers=DUMP_PAIR["workers"], shuffle=False, map_mode=DUMP_PAIR["map_mode"], pixels=False)
        assert ds2.missing == 0, f"{ds2.missing} frames of val_clean have no stored lane GT (the dump asserts it too)"
        by_row = {}
        for b in dl2:                                                                                      # pass 2: the pair model on its dump's loader; a frame is kept when B's pass saw it
            with torch.no_grad():
                _, _, o = mp(b["x"].to(dev).float(), gt=None)
            for i in range(len(b["row"])):
                if int(b["row"][i]) in b_side and int(b["row"][i]) not in by_row:
                    by_row[int(b["row"][i])] = frame_dict(o, i, ends)
            if all(r in by_row for r in rows):
                break
    finally:
        LD.QData.DEFAULT_JUNCTION = old_junction
'''),
 ('''    def make(bs, coded):
        out = []
        for a in range(0, len(rows), bs):
            r = rows[a:a + bs]''', '''    reverse = []                                                                                           # the pair loader hands the frames out in REVERSED order (the join must not depend on it)

    def make(bs, coded):
        out = []
        order = rows[::-1] if (coded and reverse) else rows
        for a in range(0, len(order), bs):
            r = order[a:a + bs]'''),
 ('''        return NS(missing=0), make(bs, True)''', '''        return NS(missing=0), make(bs, len(calls) == 2)'''),
 ('''        assert pcalls == [4, 2], pcalls ''', '''        assert (QData.DEFAULT_ENDS, QData.DEFAULT_JUNCTION) == (False, False), "both loader flags are put back after the pass"
        reverse.append(1)
        calls.clear()
        recs_r = live_records(0, "val_clean", 0.1, 5, device="cpu", pair_run="e324_s_s0", ends="att", faithful=True)
        assert [r["row"] for r in recs_r] == rows[:5] and sorted(seen[-5:]) == sorted((float(r), float(r), float(r), r, "att", 100.0 + r) for r in rows[:5]) and [s[3] for s in seen[-5:]] == rows[:5], seen[-5:]      # a reversed pair loader joins the same way
        reverse.clear()
        calls.clear()
        pcalls[:] = [4, 2]
        calls.extend([b_call, p_call])
        assert pcalls == [4, 2], pcalls '''),
])

# ---- jwire_check.py: --bs with --faithful refused, deviation report, usage line
edit("evaluation/jwire_check.py", [
 ('v.add_argument("--bs", type=int, default=4, help="the batch size of the live pass (default 4 as E325; the stored dumps were made with 8: E358 compares at --bs 8)")', 'v.add_argument("--bs", type=int, default=None, help="the batch size of the live pass (default 4 as E325; E358 attempt 3 used 8); not with --faithful, which fixes the batch sizes of the two dumps")'),
 ("ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends, bs=a.bs, faithful=a.faithful)", "if a.faithful and a.bs is not None:\n        print(\"--bs and --faithful exclude each other: the faithful pass fixes the batch sizes of the two stored dumps (jwire_data.DUMP_B / DUMP_PAIR)\")\n        return 2\n    a.bs = 4 if a.bs is None else a.bs\n    ok, n, diff = live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends, bs=a.bs, faithful=a.faithful)"),
 ("[--device cpu] [--bs 4|8]", "[--device cpu] [--bs 4|8 | --faithful]"),
 ("    live = JD.live_records(seed, \"val_clean\", 0.1, nmax, workers, bs=bs, device=dev, pair_run=pair_run, ends=ends, faithful=faithful)\n    return compare_records(live, ref)", "    live = JD.live_records(seed, \"val_clean\", 0.1, nmax, workers, bs=bs, device=dev, pair_run=pair_run, ends=ends, faithful=faithful)\n    res = compare_records(live, ref)\n    if not res[0]:\n        print(\"deviations of the frames that differ (largest absolute difference per field, DIAGNOSTIC ONLY: the rule is the tolerance of the module docstring): \" + str(deviation_report(live, ref, res[2])))\n    return res"),
 ("def compare_records(live, ref, atol=ATOL):", '''def deviation_report(live, ref, diff):
    """-> {row: {field: largest absolute difference (shape mismatch: 'shape a vs b')}} of the frames named in `diff` (a diagnostic printed with a REFUSED check; it decides nothing)"""
    out = {}
    byrow = {r["row"]: r for r in live}
    for row in diff:
        a, b = byrow.get(row), ref.get(row)
        if a is None or b is None:
            continue
        d = {}
        for k in FIELDS:
            x, y = np.asarray(a[k], float), np.asarray(b[k], float)
            if x.shape != y.shape:
                d[k] = f"shape {x.shape} vs {y.shape}"
            elif x.size and not np.allclose(x, y, atol=ATOL, equal_nan=True):
                d[k] = float(np.nanmax(np.abs(x - y)))
        out[row] = d
    return out


def compare_records(live, ref, atol=ATOL):'''),
 ("            print(\"jwire_check selftest ok\")" if False else "    print(\"jwire_check selftest ok\")", '''    # deviation report: the size of a planted difference, a shape mismatch is named, equal frames are empty
    la = dict(row=7, **{k: np.zeros((2, 2)) for k in FIELDS})
    lb = dict(la, pairs=np.array([[0.0, 0.0], [0.0, 0.0125]]))
    lc = dict(la, ends=np.zeros((3, 2)))
    rep = deviation_report([lb, lc], {7: la}, {7: ["pairs"]})
    assert rep == {7: {"pairs": 0.0125}} and deviation_report([lb], {7: la}, {}) == {}, rep
    assert deviation_report([lc], {7: la}, {7: ["ends"]}) == {7: {"ends": "shape (3, 2) vs (2, 2)"}}
    print("jwire_check selftest ok")'''),
])
print("ok")
