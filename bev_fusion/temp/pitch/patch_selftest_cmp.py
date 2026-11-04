p = "selftest_comparison.py"
s = open(p, encoding="utf-8").read()
old = '''        assert M.tolerant_counts(p, g, brute_near(np.ones_like(g), g, tol), near_p, valid).tolist() == want, "the counting step with dilations given"
'''
new = old + '''        assert (M.dilate_loop(g, tol) == M.dilate(g, tol)).all(), "the OpenCV dilation and the plain loop give the same map (the border included)"
        sels = [valid, ~valid, np.ones_like(valid), rng.random(valid.shape) > 0.5]
        many = M.tolerant_counts_many(p, g, [(near_g, near_p), (g, p)], sels)           # two tolerances (the second is tolerance 0: the maps themselves) x four selections
        assert many.shape == (2, 4, 4) and many.dtype == np.int64
        for t, (ng, np_) in enumerate(((near_g, near_p), (g, p))):
            for i, v in enumerate(sels):
                assert many[t, i].tolist() == M.tolerant_counts(p, g, ng, np_, v).tolist(), (tol, t, i, "one matrix product = the counts one selection at a time")
'''
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
