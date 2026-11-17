def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


# ---- test_qgraph_readout: the exempt (x) and Matrix-NMS (mnms) specs, promoted from the scratch test of 2026-11-11 (E288 / E296)
p = "evaluation/tests/test_qgraph_readout.py"
sub(p, "def main():\n    cases()\n", '''def exempt_and_mnms():
    """the special-type exemption (`x`, `x<p>`) and the Matrix-NMS spec (E288 / E296): a flagged node survives the close-node rule and the Matrix-NMS decay; the spec names are stable; `plain` with `x` and sigma <= 0 are refused"""
    s = RO.parse("close:1.5x@0.5")
    assert s.exempt and s.kind == "close" and RO.name(s) == "close1.5x_t0.5", s
    s2 = RO.parse("mnms:1.75s0.1x@0.5")
    assert s2.exempt and s2.sigma == 0.1 and RO.name(s2) == "mnms1.75s0.1x_t0.5"
    assert not RO.parse("close:1.5@0.5").exempt
    pos = np.array([[0, 0], [0.4, 0], [5, 5]], float)
    e = np.array([0.95, 0.9, 0.8])
    flags = np.zeros((3, 8))
    flags[1, 1] = 0.9                                                                   # node 1 is an exit
    pred = dict(nodes=dict(pos=pos, e=e, flags=flags))
    assert RO.apply(pred, RO.parse("close:1.5@0.5"))["nodes"]["e"][1] == 0.0
    assert RO.apply(pred, RO.parse("close:1.5x@0.5"))["nodes"]["e"][1] == 0.9
    assert RO.apply(pred, RO.parse("mnms:1.75s0.1@0.5"))["nodes"]["e"][1] == 0.0
    assert RO.apply(pred, RO.parse("mnms:1.75s0.1x@0.5"))["nodes"]["e"][1] == 0.9
    sp = RO.parse("close:1.5x0.95@0.5")
    assert sp.exempt == 0.95 and RO.name(sp) == "close1.5x0.95_t0.5", sp
    assert RO.apply(pred, sp)["nodes"]["e"][1] == 0.0                                 # the exit flag 0.9 is below 0.95: not protected
    for bad in ("plain:0.5x", "mnms:1.75s0@0.5"):
        try:
            RO.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"spec {bad!r} must be refused")


def main():
    exempt_and_mnms()
    cases()
''')

# ---- jgt: the rank sign (rank 0 = leftmost), a crossed pair of connectors breaks the order rule, the heat sits on the end
p = "training/grid/jgt.py"
sub(p, '''    zn, heat, head = fields(jg)
    assert zn[0].sum() > 50''', '''    for e, r in zip(jg["ent"], jg["ent_rank"]):                                           # looking north (+y) the left is -x: rank 0 is the lane at x = -1.75
        assert (r == 0) == (e[0] < 0), (e, r)
    # crossed connectors (the left entry goes to the right exit and the right entry to the left exit): the order rule is violated on this maneuver and a conflict core exists
    pcs2 = []
    for i, (s, nth) in enumerate(zip(south, north[::-1])):
        conn = run(s[-1], nth[0], 16)
        pts = np.vstack([s, conn, nth])
        pcs2.append(dict(piece=f"q{i}", pts=pts, tier=np.full(len(pts), 3), segments=[dict(kind="lane", i0=0, i1=len(s) - 1), dict(kind="connector", i0=len(s), i1=len(s) + len(conn) - 1),
                                                                            dict(kind="lane", i0=len(s) + len(conn), i1=len(pts) - 1)]))
    jx = junction_gt(dict(pieces=pcs2))
    assert jx["order"]["multi"] == 1 and jx["order"]["monotone"] == 0 and jx["order"]["identity"] == 0, jx["order"]
    assert jx["core"] is not None and not jx["core"].is_empty and jx["core"].area > 1.0
    zn, heat, head = fields(jg)
    for k, EE in enumerate((jg["ent"], jg["ext"])):                                       # the heat peaks at the cell that holds the end (row = x, column = y, origin -50 m)
        for e in EE:
            i, j = int((e[0] + 50.0) // 1.0), int((e[1] + 50.0) // 1.0)
            assert heat[k][i, j] > 0.7, (k, e, heat[k][i, j])
    assert zn[0].sum() > 50''')
print("ok")
