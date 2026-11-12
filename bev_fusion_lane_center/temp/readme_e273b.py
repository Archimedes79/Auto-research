p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()
anchor = "| `chain_eval.py` |"
rows = """| `qgraph_eval.py` | local scorer of the E273 lane head: `readout` (the named output step: e >= 0.5, p >= 0.5, one piece per section, connector pieces for links between sections), `graph_metrics` (the lane suite's own functions against the old graph GT: Q, F1@1.0, lanes found, topo F1, junction F1, duplicates, pieces per GT piece, id switches; reproduces the stored suite run of recall_sp0_dp exactly), `node_metrics`, `assign_metrics` (lane_acc, dustbin, purity, road level), `val_frames`, `score_frames`, `paired`; self-test + [tests/test_qgraph_eval.py](tests/test_qgraph_eval.py) (oracle, degradations, mutants, reproduction of d; about 4 min, CPU) |
| `qgraph_run_eval.py` | full scoring of a trained lane head on the 873 VAL frames: p (graph, nodes, assignment with the exact map and a 1-2 m map error), the control g (E272 chain + geometric assigner on the same map), the reference d, the paired scene bootstrap, the verdict by the card's rule; `qgraph_run_eval.py score --run e273_p_s0 --seed 0`, `verdict --runs ...`; writes `runs/e273/<run>/card.json` and `results.md` |
"""
assert anchor in s
s = s.replace(anchor, rows + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
p = "tools/README.md"
s = open(p, encoding="utf-8").read()
anchor = "| `viz/vertex_report.py` |"
row = "| `viz/qgraph_view.py` | pictures and videos of the E273 lane head on real VAL frames: the six cameras with the GT lane graph projected, 1 m centre pixels, 2 m road / intersection and 0.5 m dividers / boundary (GT grey underlay, correct dark, false positives light red, misses light blue), nodes by type with the Hermite links and the false / missed tints, ids (GT pieces in clean colours, delivered nodes in the expected colour of their map assignment so that an unsure assignment blends). `qgraph_view.py frame <token> --out DIR [--pred oracle|degraded]`, `scene <VAL scene> --out DIR --video`, `--selftest`; `pred_fn=qpredict.make_pred_fn(run)` for a trained run; CPU is enough |\n"
assert anchor in s
s = s.replace(anchor, row + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
p = "tools/maintenance/run_all_tests.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('run "E273 targets: the oracle on real lane GT frames" grid/test_qtargets.py\n', 'run "E273 targets: the oracle on real lane GT frames" grid/test_qtargets.py\nrun "E273 local scorer (oracle, degradations, lane suite reproduction)" test_qgraph_eval.py\nrun "E273 pictures: selftest" viz/qgraph_view.py --selftest\n')
open(p, "w", encoding="utf-8").write(s)
print("ok")
