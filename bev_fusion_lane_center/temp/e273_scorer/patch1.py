p = "evaluation/tests/test_qgraph_eval.py"
s = open(p, encoding="utf-8").read()
a = s.index("def test_shift_nodes")
b = s.index("def test_drop_nodes_and_cut_links")
new = '''def shifted(preds, d):
    out = []
    for p in preds:
        q = clone(p)
        q["nodes"]["pos"] = q["nodes"]["pos"] + [d, 0.0]
        q["links"]["curves"] = [c + [d, 0.0] for c in q["links"]["curves"]]
        out.append(q)
    return out


def test_shift_nodes():
    fr, preds = frames()
    base = score(preds, fr)
    res = {}
    for d in (0.5, 1.0, 1.5):                                              # across the lanes (they run along y); beyond 2 m a node matches the next lane
        res[d] = score(shifted(preds, d), fr)
        show(f"nodes shifted {d} m (across)", dict(res[d]["nodes"], **res[d]["graph"]), ("pos_err_med", "node_F1", "F1@0.5", "F1@1.0", "F1@1.5", "topo_F1"))
        assert abs(res[d]["nodes"]["pos_err_med"] - d) < 1e-9, "the position error of matched nodes is the shift"
    g0 = base["graph"]
    assert res[0.5]["graph"]["F1@1.0"] > g0["F1@1.0"] - 0.01 and res[0.5]["graph"]["F1@0.5"] < g0["F1@0.5"] - 0.3, "0.5 m: inside 1.0 m, outside 0.5 m"
    f = [g0["F1@1.0"]] + [res[d]["graph"]["F1@1.0"] for d in (0.5, 1.0, 1.5)]
    assert f[1] > f[2] + 0.2 > f[3] + 0.4 - 0.2 and f[2] < f[0] - 0.25 and f[3] < f[2] - 0.1, f
    assert res[0.5]["nodes"]["node_F1"] > 0.99 and res[1.0]["nodes"]["node_F1"] > 0.9, "node F1 (tolerance 2 m) holds up to 1 m"
    assert res[1.0]["graph"]["topo_F1"] < g0["topo_F1"] - 0.2


'''
s = s[:a] + new + s[b:]
s = s.replace('''            halves = []
            for i in (a, b):
                m = np.flatnonzero(nt["piece"] == i)
                halves.append(m[np.argsort(nt["j"][m])][len(m) // 2:])
''', '''            halves, h = [], min((nt["piece"] == a).sum(), (nt["piece"] == b).sum()) // 2
            for i in (a, b):
                m = np.flatnonzero(nt["piece"] == i)
                halves.append(m[np.argsort(nt["j"][m])][-h:])
''')
s = s.replace('''    assert d["lane_acc"] == 0.0 and d["dust_P"] == 0.0 and d["purity"] < 0.1
''', '''    assert d["lane_acc"] == 0.0 and d["dust_P"] == 0.0 and d["purity"] == 1.0, "purity is consistency: a lane wholly on one (wrong) piece is pure"
    assert r["purity"] < 1.0 - 0.1
''')
s = s.replace('''    pr = QE.paired(ra, rb, fr, B=30)''', '''    pr = QE.paired(ra, rb, fr, B=30)''')
s = s.replace("print(f\"   VAL tokens with a stored lane GT: {share:.4f} of 873 (signature of this checkout's autogt source)\")",
              "print(f\"   share of the 873 VAL tokens with a stored lane GT: {share:.4f} (signature of this checkout's autogt source)\")")
open(p, "w", encoding="utf-8").write(s)
