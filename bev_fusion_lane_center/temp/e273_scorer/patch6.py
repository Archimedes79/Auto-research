p = "evaluation/tests/test_qgraph_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    assert g["F1@1.0"] >= 0.99 and g["P@1.0"] > 0.999 and g["false_len_frac"] < 0.001, "the oracle graph must lie on the old graph GT"
''', '''    assert g["F1@1.0"] >= 0.99 and g["P@1.0"] > 0.999 and g["false_len_frac"] < 0.001, "the oracle graph must lie on the old graph GT"
    assert g["topo_F1"] > 0.95 and g["junction_F1"] > 0.9 and g["R@1_far"] > 0.97, "...and carry its connectivity (the representation's own ceiling against the old GT)"
''')
new = '''def test_readout_structure():
    """the rules of the output step: a piece lies in ONE section, every kept link between two sections is a connector piece, an edge joins the end node of a piece to the start node of the next,
    thresholds and kept nodes decide what is delivered"""
    fr, preds = frames(step=43)
    for p in preds:
        graph, bk = QE.readout(p)
        sec, src, dst = p["nodes"]["section"], p["links"]["src"], p["links"]["dst"]
        L = bk["links"]
        assert len(L) == len(src), "the oracle keeps every link"
        cross = sec[src] != sec[dst]
        assert sum(k == "link" for k in bk["kind"]) == cross.sum() > 0 and list(bk["cross"]) == list(cross[L])
        for k, n, s_ in zip(bk["kind"], bk["piece_nodes"], bk["section"]):
            if k == "section":
                assert set(sec[n]) == {s_}, "a section piece holds one section"
        n_in = sum(len(l) for k, l in zip(bk["kind"], bk["piece_links"]) if k == "section")
        assert n_in == (~cross).sum(), "every link inside a section is on a section piece"
        assert all(graph.nodes[i][-1] == graph.nodes[j][0] for i, j in graph.edges) and len(graph.edges) > 0, "an edge goes from the end node of a piece to the start node of the next"
        assert all(len(set(map(tuple, np.round(l, 6))) ) > 1 for l in graph.lines), "no piece of zero length"
        # thresholds: a link below thr_link or an end node below thr_e is not delivered
        q = clone(p)
        q["links"]["p"][:] = 0.4
        assert len(QE.readout(q)[0].lines) == 0 and len(QE.readout(q, thr_link=0.3)[0].lines) == len(graph.lines)
        q = clone(p)
        q["nodes"]["e"][:] = 0.4
        assert len(QE.readout(q)[0].lines) == 0 and len(QE.readout(q, thr_e=0.3)[0].lines) == len(graph.lines)


'''
s = s.replace("def test_hermite_path_flags_and_directions", new + "def test_hermite_path_flags_and_directions", 1)
s = s.replace('("oracle through the scorer", test_oracle),', '("oracle through the scorer", test_oracle), ("readout structure", test_readout_structure),')
open(p, "w", encoding="utf-8").write(s)
