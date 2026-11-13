p = "tools/maintenance/run_all_tests.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('run "E273 lane head: all phases, gradients" grid/qgraph.py\n', 'run "E273 lane head: all phases, gradients" grid/qgraph.py\nrun "E273 pixel targets (synthetic)" grid/qraster.py\nrun "E273 pixel targets on real lane GT frames" grid/test_qraster.py\nrun "E273 targets: the oracle on real lane GT frames" grid/test_qtargets.py\n')
open(p, "w", encoding="utf-8").write(s)
