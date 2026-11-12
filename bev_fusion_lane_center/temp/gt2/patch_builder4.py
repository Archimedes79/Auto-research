p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
marker = "exec(open(os.path.join(HERE, \"report_text.py\"), encoding=\"utf-8\").read())"
funcs = '''def snap_table():
    """E311: the snapped connectors, every run / threshold scored by temp/gt2/snap_score.py (differences to the plain graph of the same seed); next to them the same pairs drawn from the keypoints (E309 sweep or the card of the run) when stored"""
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "..", "gt2", "snap", "*.json"))):
        d = J(f)
        if not d:
            continue
        n, b = d["numbers"]["head"], d["numbers"]["base"]
        un = THR.get((d["run"], d["thr"]))
        if un is None:
            c = J(f"runs/{d['run']}/ends_card.json")
            un = c if c and abs(c.get("thr", 0.5) - d["thr"]) < 1e-9 else None
        un_t = sg(un["numbers"]["head"]["topo_F1"] - un["numbers"]["base"]["topo_F1"]) if un else "-"
        rows.append([d["run"], f"{d['thr']:.1f}", str(d["curves"]), f3(n["topo_F1"]), sg(n["topo_F1"] - b["topo_F1"]), un_t, sg(n["lanes_found"] - b["lanes_found"]), sg(n["junction_F1"] - b["junction_F1"]), sg(n["junction_R"] - b["junction_R"]),
                     sg(n["dup_frac"] - b["dup_frac"]), sg(n["F1@1.0"] - b["F1@1.0"]), sg(n["vert100_dp"] - b["vert100_dp"])])
    rows.sort(key=lambda r: (r[0], -float(r[1])))
    if not rows:
        return "<p>pending</p>"
    return tbl(["run", "p >=", "curves", "topo F1", "vs plain graph", "(connectors from the keypoints)", "lanes found", "junction F1", "junction recall", "dup_frac", "F1@1.0", "vertices / 100 m"], rows, num=range(1, 12))


'''
assert marker in s
s = s.replace(marker, funcs + marker, 1)
old = '<h3>E309: the delivery threshold of the pairs (differences to the plain delivered graph)</h3>{thr_table()}{PAIRS4}'
assert old in s
s = s.replace(old, old + '<h3>E311: snap the connector ends to the delivered lane pieces (differences to the plain delivered graph)</h3>{snap_table()}{PAIRS5}', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
