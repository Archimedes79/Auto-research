p = "evaluation/tests/test_qgraph_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import copy\nimport json\nimport sys", "import sys")
s = s.replace("import qgraph_eval as QE\nimport topo_metrics as TM\n", "import qgraph_eval as QE\n")
s = s.replace("    import numpy as _np\n    sys.path.insert(0, BP.REPO", "    sys.path.insert(0, BP.REPO")
s = s.replace("_np.", "np.")
# new checks
new = '''def test_hermite_path_flags_and_directions():
    fr, preds = frames()
    base = score(preds, fr)
    ps, pf, pd = [], [], []
    for p in preds:
        q = clone(p)
        q["links"]["curves"] = None                                        # no stored curves: the Hermite curves of the node poses and the handles (1/3, 1/3)
        ps.append(q)
        q = clone(p)
        q["nodes"]["flags"][:, QE.QT.FI["split"]] = 0.0                    # no split flag anywhere
        pf.append(q)
        q = clone(p)
        c, s_ = np.cos(np.radians(20)), np.sin(np.radians(20))
        q["nodes"]["dirv"] = q["nodes"]["dirv"] @ np.array([[c, s_], [-s_, c]])   # every direction turned by 20 degrees
        pd.append(q)
    h, f, d = score(ps, fr), score(pf, fr), score(pd, fr)
    show("oracle / Hermite handles 1/3", dict(h["graph"], f1_oracle=base["graph"]["F1@1.0"]), ("f1_oracle", "F1@1.0", "topo_F1", "junction_F1"))
    assert h["graph"]["F1@1.0"] > base["graph"]["F1@1.0"] - 0.03 and h["graph"]["topo_F1"] > base["graph"]["topo_F1"] - 0.03, "chord / 3 handles reproduce the curves"
    assert f["nodes"]["ev_split_R"] == 0.0 and f["nodes"]["ev_split_P"] == 0.0 and f["nodes"]["ev_merge_R"] == 1.0 and f["nodes"]["node_F1"] == 1.0
    assert abs(d["nodes"]["dir_err_med"] - 20.0) < 1e-6 and d["nodes"]["pos_err_med"] == 0.0


'''
s = s.replace("def test_drop_nodes_and_cut_links", new + "def test_drop_nodes_and_cut_links", 1)
s = s.replace('("nodes dropped, links cut", test_drop_nodes_and_cut_links)', '("Hermite path, flags, directions", test_hermite_path_flags_and_directions),\n          ("nodes dropped, links cut", test_drop_nodes_and_cut_links)')
open(p, "w", encoding="utf-8").write(s)
