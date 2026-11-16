p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''        ka, kb = np.intersect1d(ca["links"]["key"], cb["links"]["key"], return_indices=True), None
        _, ia, ib = ka
''', '''        _, ia, ib = np.intersect1d(ca["links"]["key"], cb["links"]["key"], return_indices=True)
''')
open(p, "w", encoding="utf-8").write(s)
