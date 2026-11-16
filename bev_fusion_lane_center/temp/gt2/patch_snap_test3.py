p = "evaluation/qends_eval.py"
s = open(p, encoding="utf-8").read()
old = "    # snap_guard: two abutting pieces"
assert old in s
new = '''    rg, _, wg = add_connectors(dl, [], [[i] for i in range(7)], dk, dlk, dm, pr, snap=True, snap_nogeom=True)
    assert np.allclose(rg["pts"][-1][0], (0.9, -2.3)) and sorted(e for e in rg["edges"] if 7 in e) == [(3, 7), (7, 5)] and wg == 1, (rg["pts"][-1][0], rg["edges"])          # snap_nogeom: the keypoints stay the curve ends, one piece per end is wired
'''
s = s.replace(old, new + old, 1)
open(p, "w", encoding="utf-8").write(s)
