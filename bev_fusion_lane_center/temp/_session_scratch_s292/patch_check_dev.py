import io
p = "evaluation/jwire_check.py"
s = io.open(p, encoding="utf-8").read()
a = '''    lb = dict(la, pairs=np.array([[0.0, 0.0], [0.0, 0.0125]]))
    lc = dict(la, ends=np.zeros((3, 2)))
    rep = deviation_report([lb, lc], {7: la}, {7: ["pairs"]})
    assert rep == {7: {"pairs": 0.0125}} and deviation_report([lb], {7: la}, {}) == {}, rep
    assert deviation_report([lc], {7: la}, {7: ["ends"]}) == {7: {"ends": "shape (3, 2) vs (2, 2)"}}'''
assert s.count(a) == 1
b = '''    la8 = dict(la, row=8)
    lb = dict(la, pairs=np.array([[0.0, 0.0], [0.0, 0.0125]]))
    lc = dict(la8, ends=np.zeros((3, 2)))
    rep = deviation_report([lb, lc], {7: la, 8: la8}, {7: ["pairs"], 8: ["ends"]})
    assert rep == {7: {"pairs": 0.0125}, 8: {"ends": "shape (3, 2) vs (2, 2)"}}, rep
    assert deviation_report([lb], {7: la}, {}) == {} and deviation_report([lb], {7: la}, {9: ["pairs"]}) == {}'''
s = s.replace(a, b)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
