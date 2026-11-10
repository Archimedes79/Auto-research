p = "evaluation/tests/test_qgraph_eval.py"
s = open(p, encoding="utf-8").read()
s = s.replace("on REAL VAL frames. CPU only (about 3 minutes).", "on REAL VAL frames. CPU only (about 4 minutes).")
s = s.replace('''  * degradations move the numbers''', '''  * readout rules: a piece lies in one section, every kept link between two sections is a connector piece, edges join end node to start node, thresholds decide what is delivered;
  * degradations move the numbers''')
s = s.replace("import sys\nimport time\nimport traceback\n", "import os\nimport sys\nimport time\nimport traceback\n")
s = s.replace("        import os\n        sys.path.insert(0, BP.REPO)", "        sys.path.insert(0, BP.REPO)")
s = s.replace("    assert abs(one[\"F1@1.0\"] - QE.graph_numbers([rec[\"graph\"][3]])[\"F1@1.0\"]) < 1e-12\n    return out\n", "    assert abs(one[\"F1@1.0\"] - QE.graph_numbers([rec[\"graph\"][3]])[\"F1@1.0\"]) < 1e-12\n")
open(p, "w", encoding="utf-8").write(s)
