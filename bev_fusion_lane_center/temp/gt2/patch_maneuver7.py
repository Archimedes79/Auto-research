p = "temp/jr/maneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('ALL = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle", "oracle+fit"]', 'ALL = ["plain", "p>=0.5", "p>=0.2", "p>=0.1", "oracle", "oracle+fit", "p>=0.5 snap", "p>=0.1 snap", "oracle snap", "oracle+fit snap"]')
sub("import qjunction as QJ  # noqa: E402", "import qjunction as QJ  # noqa: E402\nimport snapconn as SC  # noqa: E402")
old_start = s.index("    for name, lg, thr in [x for x in")
old_end = s.index("    return out\n", old_start)
new = '''    table = (("p>=0.5", e["logit"], 0.5), ("p>=0.2", e["logit"], 0.2), ("p>=0.1", e["logit"], 0.1), ("oracle", logit_or, 0.5), ("oracle+fit", logit_or, 0.5),
             ("p>=0.5 snap", e["logit"], 0.5), ("p>=0.1 snap", e["logit"], 0.1), ("oracle snap", logit_or, 0.5), ("oracle+fit snap", logit_or, 0.5))
    for name, lg, thr in [x for x in table if x[0] in VARIANTS]:
        snap = name.endswith(" snap")
        base = name[:-5] if snap else name
        p = 1.0 / (1.0 + np.exp(-lg)) * e["allowed"]
        ia, ib = np.nonzero(p >= thr)
        prs = [(e["kp_pos"][i], e["kp_dir"][i], e["kp_pos"][k + j], e["kp_dir"][k + j]) + (scales_or[(int(i), int(j))] if base == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and base.startswith("p")) else ())) for i, j in zip(ia, ib)]
        add = SC.add_connectors_snap if snap else QEV.add_connectors
        raw, _, _ = add(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), prs)
        out[name] = raw
'''
s = s[:old_start] + new + s[old_end:]
sub('"oracle+fit": "#d62728"}\ncols', '"oracle+fit": "#d62728", "p>=0.5 snap": "#006400", "p>=0.1 snap": "#00008b", "oracle snap": "#b8860b", "oracle+fit snap": "#8b0000"}\ncols')
open(p, "w", encoding="utf-8").write(s)
print("ok")
