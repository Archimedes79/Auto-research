p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


marker = "def snap_table():"
funcs = '''def ladder_table():
    """the ladder of the thread: the delivered graph's gain over the plain graph, step by step, every cell read from the card of the run (both seeds)"""
    def cd(path):
        return J(path)

    steps = [("E287 p2: pair head, connectors from the keypoints, p >= 0.5", "runs/e287_p2_s{s}/ends_card.json"),
             ("E305: maps and pairs trained together, same read-out", "runs/e305_e_s{s}/ends_card.json"),
             ("+ connector ends snapped, p >= 0.1 (E311)", "temp/gt2/snap/e305_e_s{s}_thr0.1.json"),
             ("E303: + learned handle scales (E303 pairs on the E286 maps)", "runs/e303_s_s{s}/ends_card_snap_thr0.1.json"),
             ("<b>E312: + the narrower end heat E307 (heat sigma 1.0 m), scales, snapped, p >= 0.1</b>", "runs/e312_p_s{s}/ends_card_snap_thr0.1.json")]
    rows = []
    for name, pat in steps:
        cs = [cd(pat.format(s=s)) for s in (0, 1)]
        if not all(cs):
            rows.append([name] + ["pending"] * 6)
            continue
        def cell(k):
            return " / ".join(sg(c["difference"][k]) for c in cs)
        rows.append([name, " / ".join(f3(c["numbers"]["head"]["topo_F1"]) for c in cs), cell("topo_F1"), cell("lanes_found"), cell("junction_F1"), cell("junction_R"), cell("dup_frac")])
    return tbl(["step (the delivered graph = B + the hard close-node step + the connectors of the pair head)", "topo F1 (seed 0 / seed 1)", "topo F1 vs the plain graph", "lanes found", "junction F1", "junction recall", "dup_frac"], rows, num=range(1, 7))


'''
sub(marker, funcs + marker)
sub("<h3>E311: snap the connector ends to the delivered lane pieces (differences to the plain delivered graph)</h3>{snap_table()}{PAIRS5}", "<h3>The ladder of the thread: what each step adds to the delivered graph</h3>{ladder_table()}{PAIRS6}<h3>E311: snap the connector ends to the delivered lane pieces (differences to the plain delivered graph)</h3>{snap_table()}{PAIRS5}")
sub('    t.append(("good", "+0.030", "topo F1 over the plain graph (E305, snapped connector ends, p >= 0.1, seed 0; seed 1 +0.029)", "the same pairs drawn from the keypoints at p >= 0.5: +0.007 (E311)"))',
    '    t.append(("good", "+0.057", "topo F1 over the plain graph (E312: narrower end heat, handle scales, snapped connector ends, p >= 0.1, seed 0; seed 1 +0.062)", "the pair head alone, connectors from the keypoints at p >= 0.5: +0.006 (E287)"))')
sub("{FAIL}{FAIL2}{maneuver_block()}<h3>After: E303 pairs with learned handle scales, connector ends snapped, p >= 0.1</h3>{errors_block('errors_snap_e303', 'aft_', 'After (E303, snapped, p >= 0.1). ')}", "{FAIL}{FAIL2}{maneuver_block()}<h3>After: E312 (narrower end heat, handle scales, snapped ends, p >= 0.1)</h3>{errors_block('errors_snap_e312', 'aft_', 'After (E312, snapped, p >= 0.1). ')}")
open(p, "w", encoding="utf-8").write(s)
print("ok")
