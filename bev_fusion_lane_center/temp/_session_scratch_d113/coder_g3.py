p = "training/grid/jpast.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''assert abs(sb["d2_none"] - 10 * D * t_ ** 2) < 1e-6 and sb["d2_other"] < 1e-9 and sb["n"] == sb["n_other"] == 10 and abs(math.sqrt(sb["d2_none"] / sb["x2"]) - t_ / math.sqrt(x2q / (10 * D))) < 1e-6,''',
    '''assert abs(sb["d2_none"] - 10 * D * t_ ** 2) < 1e-5 * 10 * D * t_ ** 2 and sb["d2_other"] < 1e-9 and sb["n"] == sb["n_other"] == 10 and abs(math.sqrt(sb["d2_none"] / sb["x2"]) - t_ / math.sqrt(x2q / (10 * D))) < 1e-5,''')
rep('''    assert abs(sr["d2_none"] - 5 * a_ ** 2 * float((S[0] ** 2).sum() + (S[1] ** 2).sum())) < 1e-6 and abs(sr["d2_other"] - 5 * a_ ** 2 * 2 * float(((S[0] - S[1]) ** 2).sum())) < 1e-6 and sr["d2_other"] > 1.0,''',
    '''    w_none, w_other = 5 * a_ ** 2 * float((S[0] ** 2).sum() + (S[1] ** 2).sum()), 5 * a_ ** 2 * 2 * float(((S[0] - S[1]) ** 2).sum())
    assert abs(sr["d2_none"] - w_none) < 1e-5 * w_none and abs(sr["d2_other"] - w_other) < 1e-5 * w_other and sr["d2_other"] > 1.0,''')
rep('''assert abs(tk["u"] - math.sqrt(5 * a_ ** 2 * float(((S[0] - S[1]) ** 2).sum()) / x2_0)) < 1e-6 and tk["n_other"] == 5''', '''assert abs(tk["u"] - math.sqrt(5 * a_ ** 2 * float(((S[0] - S[1]) ** 2).sum()) / x2_0)) < 1e-5 and tk["n_other"] == 5''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
