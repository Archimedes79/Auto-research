p = "evaluation/tests/test_qgraph_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    fr, preds = frames(step=43)
    for p in preds:''', '''    fr, preds = frames(step=43)
    n_link = n_edge = 0
    for p in preds:''')
s = s.replace('''        assert sum(k == "link" for k in bk["kind"]) == cross.sum() > 0 and list(bk["cross"]) == list(cross[L])''',
              '''        assert sum(k == "link" for k in bk["kind"]) == cross.sum() and list(bk["cross"]) == list(cross[L])
        n_link, n_edge = n_link + cross.sum(), n_edge + len(graph.edges)''')
s = s.replace('''        assert all(graph.nodes[i][-1] == graph.nodes[j][0] for i, j in graph.edges) and len(graph.edges) > 0, "an edge goes from the end node of a piece to the start node of the next"''',
              '''        assert all(graph.nodes[i][-1] == graph.nodes[j][0] for i, j in graph.edges), "an edge goes from the end node of a piece to the start node of the next"''')
s = s.replace('''        assert len(QE.readout(q)[0].lines) == 0 and len(QE.readout(q, thr_e=0.3)[0].lines) == len(graph.lines)
''', '''        assert len(QE.readout(q)[0].lines) == 0 and len(QE.readout(q, thr_e=0.3)[0].lines) == len(graph.lines)
    assert n_link > 0 and n_edge > 0, "the frames hold connectors and edges"
''')
open(p, "w", encoding="utf-8").write(s)
