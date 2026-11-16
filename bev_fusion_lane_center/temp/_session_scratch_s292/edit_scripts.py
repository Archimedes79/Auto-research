def sub(p, a, b):
    s = open(p, encoding="utf-8").read()
    assert s.count(a) == 1, (p, a[:60])
    open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
sub("tools/maintenance/run_gates.sh", 'run "head: torch Matrix-NMS layer (E292)"                 grid/qmnms.py\n',
    'run "head: torch Matrix-NMS layer (E292)"                 grid/qmnms.py\nrun "head: Matrix-NMS switch, phase 9, mutations (E292)"  grid/test_mnms_layer.py\n')
sub("tools/maintenance/run_all_tests.sh", 'run "E277 verdict rule (fake cards)" test_verdict277.py\n',
    'run "E277 verdict rule (fake cards)" test_verdict277.py\nrun "E292 Matrix-NMS layer = matrix_nms on 20 frames of B\'s real dump (kill criterion), switch, phase 9, mutations" grid/test_mnms_layer.py --real\n')
