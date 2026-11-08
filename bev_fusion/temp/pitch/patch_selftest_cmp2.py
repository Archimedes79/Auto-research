p = "../autoresearch/evalsuite/selftest_comparison.py"
s = open(p, encoding="utf-8").read()
old = '''    one = np.zeros((9, 9), bool)
    one[4, 4] = True
    assert (M.dilate(one, 0.3 / 0.1)'''
new = '''    big_p, big_g = rng.random((120, 104)) > 0.3, rng.random((120, 104)) > 0.3                     # the BEV grid: counts of thousands of cells (float16 would round them)
    big_v = [np.ones_like(big_p), rng.random(big_p.shape) > 0.3]
    bn = [(M.dilate(big_g, 2), M.dilate(big_p, 2))]
    assert M.tolerant_counts_many(big_p, big_g, bn, big_v)[0].tolist() == [M.tolerant_counts(big_p, big_g, bn[0][0], bn[0][1], v).tolist() for v in big_v], "exact on a full grid"
    one = np.zeros((9, 9), bool)
    one[4, 4] = True
    assert (M.dilate(one, 0.3 / 0.1)'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
