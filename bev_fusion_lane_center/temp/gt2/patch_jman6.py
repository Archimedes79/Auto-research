p = "evaluation/jmaneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('    thr = {v: float(v[1:]) for v in names if v.startswith("p") and v[1:].replace(".", "", 1).isdigit()}',
    '    snapped = {v for v in names if v.startswith("p") and v.endswith("s") and v[1:-1].replace(".", "", 1).isdigit()}                    # p0.1s = connectors at p >= 0.1 with snapped ends (E311)\n    thr = {v: float(v[1:-1] if v in snapped else v[1:]) for v in names if v in snapped or (v.startswith("p") and v[1:].replace(".", "", 1).isdigit())}')
sub("raw, _, _ = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], long_mask, prs)", "raw, _, _ = QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], long_mask, prs, snap=v in snapped)")
sub("`p<t>` (written `p0.5`, shell-safe;", "`p<t>` (written `p0.5`, shell-safe; `p0.5s`: with snapped connector ends, E311;")
open(p, "w", encoding="utf-8").write(s)
print("ok")
