"""one-off: 5-field items (second-gap steps) in evaluate / picture of tests/test_flow_explains.py; moving / static EPE"""
p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("    for maps, vgt, bid, steps in test:\n", "    for maps, vgt, bid, steps, steps12 in test:\n")
rep('''               "extrap": propagate(m[:, 1], [steps], carry(v, [steps]), G)[0],
               "extrap GT v": propagate(m[:, 1], [steps], carry(vg, [steps]), G)[0]}''',
    '''               "extrap": propagate(m[:, 1], [steps12], carry(v, [steps]), G)[0],
               "extrap GT v": propagate(m[:, 1], [steps12], carry(vg, [steps]), G)[0]}''')
rep('''                 "vehicle interiors": inner, "vehicle edges": veh & ~inner}''',
    '''                 "vehicle interiors": inner, "vehicle edges": veh & ~inner,
                 "vehicles moving (> 1 m/s)": veh & (vgt.norm(dim=0) > 1.0),
                 "vehicles static (< 0.2 m/s)": veh & (vgt.norm(dim=0) < 0.2)}''')
rep("        maps, vgt, bid, steps = test[r]\n", "        maps, vgt, bid, steps, steps12 = test[r]\n")
rep('''               overlay(propagate(m[:, 1], [steps], carry(v, [steps]), G)[0][0].cpu(), maps[2]),''',
    '''               overlay(propagate(m[:, 1], [steps12], carry(v, [steps]), G)[0][0].cpu(), maps[2]),''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
