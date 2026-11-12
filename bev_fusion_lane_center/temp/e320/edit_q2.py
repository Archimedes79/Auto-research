def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


edit("temp/e320/dev/qends_eval.pysrc", [
    ('''def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs,''', '''def make_pairs(e, ia, ib, k, attach=False, chord=False, fit=None):
    """the `pairs` of `add_connectors` for the candidates (ia[n], ib[n]) of a dump frame `e` (`jends_eval.py dump`; k = keypoints per kind): (entry end, entry heading, exit end, exit heading) + the learned handle scales (pr[4], pr[5]) when the dump has them and not `chord`
    (`fit`: a dict {(i, j): scales} instead, the oracle with fitted scales of `jmaneuver.py`). attach (E320): the ATTACHED ends and headings of the dump (att_pos / att_dir) instead of the keypoints'; the one place where the read-outs (cmd_score, jmaneuver) choose between the two."""
    qp, qd = (e["att_pos"], e["att_dir"]) if attach else (e["kp_pos"], e["kp_dir"])
    if fit is not None:
        return [(qp[i], qd[i], qp[k + j], qd[k + j]) + fit.get((int(i), int(j)), ()) for i, j in zip(ia, ib)]
    return [(qp[i], qd[i], qp[k + j], qd[k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None and not chord else ()) for i, j in zip(ia, ib)]


def add_connectors(lines, edges, nodes, kinds, links, long_mask, pairs,'''),
    ('''        qp, qd = (e["att_pos"], e["att_dir"]) if attach else (e["kp_pos"], e["kp_dir"])                                  # E320: the attached ends of the dump instead of the keypoints
        pairs = [(qp[i], qd[i], qp[k + j], qd[k + j]) + (tuple(e["scale"][i, j]) if e.get("scale") is not None and not a.chord else ()) for i, j in zip(ia, ib)]
''', '''        pairs = make_pairs(e, ia, ib, k, attach=attach, chord=a.chord)                                                      # E320: attach = the attached ends of the dump instead of the keypoints
        le = np.array([np.asarray(P, float)[-1] for P in graph.lines if len(P)]).reshape(-1, 2)                              # critic 2026-11-12 (C3): do the connector ends touch a delivered piece end (an end of a line, within 1 m)?
        fs = np.array([np.asarray(P, float)[0] for P in graph.lines if len(P)]).reshape(-1, 2)
        near = lambda pts, q: bool(len(pts) and np.linalg.norm(pts - q, axis=1).min() <= 1.0)  # noqa: E731
        for i, j in zip(ia, ib):
            gap["n"] += 1
            gap["in"] += near(le, e["att_pos"][i] if attach else e["kp_pos"][i])
            gap["out"] += near(fs, e["att_pos"][k + j] if attach else e["kp_pos"][k + j])
            gap["kp_in"] += near(le, e["kp_pos"][i])
            gap["kp_out"] += near(fs, e["kp_pos"][k + j])
'''),
    ('''        n_curves += n_c
        n_wired += n_w
        rec["head"].append(QE.graph_counts(raw, f))''', '''        n_curves += n_c
        n_wired += n_w
        n_drop += len(graph.lines) - (len(raw["pts"]) - len(pairs))                                                          # B's long links that a connector covers and the read-out drops
        rec["head"].append(QE.graph_counts(raw, f))'''),
    ('''    n_curves = n_wired = 0
    t0 = time.time()''', '''    n_curves = n_wired = n_drop = 0
    gap = dict(n=0, **{k_: 0 for k_ in ("in", "out", "kp_in", "kp_out")})
    t0 = time.time()'''),
    ('''    snap = bool(getattr(a, "snap", False))
    out["snap"] = snap
    out["attach"] = attach''', '''    snap = bool(getattr(a, "snap", False))
    out["snap"] = snap
    out["attach"] = attach
    out["long_links_dropped"] = n_drop
    r_ = lambda x: float(x) / max(gap["n"], 1)  # noqa: E731
    out["ends_within_1m_of_a_delivered_piece_end"] = dict(pairs=gap["n"], entry=r_(gap["in"]), exit=r_(gap["out"]), keypoint_entry=r_(gap["kp_in"]), keypoint_exit=r_(gap["kp_out"]))      # of the ends the connectors use (attached ends under attach) and of the raw keypoints'''),
    ('''    print("qends_eval selftest ok")''', '''    # E320, critic 2026-11-12: the attached ends of a dump frame reach the curves (the mutation `qp, qd = kp_pos, kp_dir` passed every earlier test), also in the primary read-out (attached ends + one nearest piece per end, the geometry kept)
    e = dict(kp_pos=np.array([[1.2, -2.5], [8.9, 2.5]]), kp_dir=np.array([[0.0, 1.0], [0.0, 1.0]]), att_pos=np.array([[0.0, -2.0], [8.0, 2.0]]), att_dir=np.array([[0.0, 1.0], [0.0, 1.0]]), scale=np.full((1, 1, 2), 0.3))
    pk, pa_ = make_pairs(e, [0], [0], 1), make_pairs(e, [0], [0], 1, attach=True)
    assert np.allclose(pk[0][0], [1.2, -2.5]) and np.allclose(pa_[0][0], [0.0, -2.0]) and np.allclose(pa_[0][2], [8.0, 2.0]) and len(pa_[0]) == 6 and len(make_pairs(e, [0], [0], 1, attach=True, chord=True)[0]) == 4
    ck = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pk)[0]["pts"][-1]
    ca = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pa_)[0]["pts"][-1]
    assert np.allclose(ck[0], [1.2, -2.5]) and np.allclose(ca[0], [0.0, -2.0]) and np.allclose(ca[-1], [8.0, 2.0]), (ck[0], ca[0])
    rp, n_p, w_p = add_connectors(lines, [(0, 2), (2, 1)], [[0], [1], [2]], kinds, links, long_mask, pa_, snap=True, snap_nogeom=True)
    assert w_p == 1 and np.allclose(rp["pts"][-1][0], [0.0, -2.0]) and (0, len(rp["pts"]) - 1) in rp["edges"] and (len(rp["pts"]) - 1, 1) in rp["edges"], rp["edges"]                      # the curve stays on the attached ends, one piece per end is wired
    assert len(make_pairs(dict(e, scale=None), [0], [0], 1)[0]) == 4 and make_pairs(e, [0], [0], 1, fit={(0, 0): (0.5, 0.6)})[0][4:] == (0.5, 0.6)
    print("qends_eval selftest ok")'''),
])

edit("temp/e320/dev/jmaneuver.pysrc", [
    ('''    attached = {v for v in names if v.startswith("p") and v.endswith("a") and v[1:-1].replace(".", "", 1).isdigit()}                    # p0.1a = connectors at p >= 0.1 from the ATTACHED ends of the dump, no snap (E320)
    thr = {v: float(v[1:-1] if (v in snapped or v in attached) else v[1:]) for v in names if v in snapped or v in attached or (v.startswith("p") and v[1:].replace(".", "", 1).isdigit())}''', '''    parsed = {v: re.fullmatch(r"p(\\d+(?:\\.\\d+)?)(s|an|a|ar(\\d+(?:\\.\\d+)?))?", v) for v in names}                                       # p<t> | p<t>s snapped | p<t>a attached, proximity wiring | p<t>an attached + one nearest piece per end (the E320 primary read-out) | p<t>ar<r> attached + wiring radius r m
    attached_n = {v for v, m in parsed.items() if m and m.group(2) == "an"}
    attached_r = {v: float(m.group(3)) for v, m in parsed.items() if m and m.group(3)}
    attached = {v for v, m in parsed.items() if m and m.group(2) in ("a", "an")} | set(attached_r)
    thr = {v: float(m.group(1)) for v, m in parsed.items() if m}'''),
    ('''                qp, qd = (e["att_pos"], e["att_dir"]) if v in attached else (e["kp_pos"], e["kp_dir"])
                prs = [(qp[i], qd[i], qp[K + j], qd[K + j]) + (scales_or.get((int(i), int(j)), ()) if v == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and not a.chord and v != "oracle") else ()))
                       for i, j in zip(ia, ib)]
                raw, _, _ = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, prs, snap=v in snapped)''', '''                prs = QEV.make_pairs(e, ia, ib, K, attach=v in attached, chord=a.chord or v == "oracle", fit=scales_or if v == "oracle+fit" else None)
                raw, _, _ = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], long_mask, prs, snap=v in snapped or v in attached_n, snap_nogeom=v in attached_n, wire_pos=attached_r.get(v))'''),
    ('''`p0.1a`: connectors from the ATTACHED ends of an E320 dump (att_pos / att_dir), no snap;''', '''`p0.1a`: connectors from the ATTACHED ends of an E320 dump (att_pos / att_dir), proximity wiring, no snap; `p0.1an`: the same ends with one nearest piece wired per end and the curve kept (the primary read-out of E320, `--attach --snap --snap-nogeom`);'''),
])
print("q2 ok")

edit("temp/e320/dev/jmaneuver.pysrc", [
    ("import pickle\n", "import pickle\nimport re\n"),
])
edit("temp/e320/dev/qends_eval.pysrc", [
    ("snap_nogeom=False, snap_both=False):\n", "snap_nogeom=False, snap_both=False, wire_pos=None):\n"),
    ("    keep = []\n    for i, k in enumerate(kinds):", "    wp = WIRE_POS if wire_pos is None else float(wire_pos)                                                     # E320: the radius (m) of the proximity wiring, default today's 4 m; the tight read-outs of the attached ends use 1.5 / 1.0\n    keep = []\n    for i, k in enumerate(kinds):"),
    ("if d <= WIRE_POS and _ang(te, ta) <= WIRE_ANG and d < d_in:", "if d <= wp and _ang(te, ta) <= WIRE_ANG and d < d_in:"),
    ("if d <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG and d < d_out:", "if d <= wp and _ang(ts, tb) <= WIRE_ANG and d < d_out:"),
    ("if np.linalg.norm(le - pa) <= WIRE_POS and _ang(te, ta) <= WIRE_ANG:", "if np.linalg.norm(le - pa) <= wp and _ang(te, ta) <= WIRE_ANG:"),
    ("if np.linalg.norm(fs - pb) <= WIRE_POS and _ang(ts, tb) <= WIRE_ANG:", "if np.linalg.norm(fs - pb) <= wp and _ang(ts, tb) <= WIRE_ANG:"),
    ("""snap_nogeom=bool(getattr(a, "snap_nogeom", False)), snap_both=bool(getattr(a, "snap_both", False)))\n        n_curves += n_c""", """snap_nogeom=bool(getattr(a, "snap_nogeom", False)), snap_both=bool(getattr(a, "snap_both", False)), wire_pos=attach_r)\n        n_curves += n_c"""),
    ("""    attach = bool(getattr(a, "attach", False))\n""", """    attach = bool(getattr(a, "attach", False))\n    attach_r = getattr(a, "attach_r", None)                                                                                  # E320: the wiring radius (m) of the attached read-out (default: today's 4 m); the card file name carries it\n    assert attach_r is None or attach, "--attach-r needs --attach"\n"""),
    ("""    out["attach"] = attach\n""", """    out["attach"] = attach\n    out["attach_r"] = attach_r\n"""),
    ("""("attach_" if attach else "") + f"thr{a.thr}.json\"""", """(("attach_" + (f"r{attach_r}_" if attach_r else "")) if attach else "") + f"thr{a.thr}.json\""""),
])
edit("temp/e320/dev/jends_eval.pysrc", [
    ("""attach=bool(getattr(a, "attach", False))))""", """attach=bool(getattr(a, "attach", False)), attach_r=getattr(a, "attach_r", None)))"""),
    ("""            s.add_argument("--attach", action="store_true",""", """            s.add_argument("--attach-r", type=float, default=None, help="E320: with --attach, the radius (m) of the proximity wiring of the connector ends (default 4: every piece end in reach; the tight read-outs use 1.5 and 1.0); the card file name carries it")
            s.add_argument("--attach", action="store_true","""),
])
print("q2b ok")
