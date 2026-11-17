import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)
rep('''    ref_rec = None if ref is None else {k: v[:len(frames)] for k, v in ref["rec"].items()}''', '''    pick = lambda rec: {k: [v[int(i)] for i in r["idx"]] for k, v in rec.items()}  # noqa: E731   # the records of plain B / B' on the frames of this read-out (the cache order = the VAL order)
    ref_rec = None if ref is None else pick(ref["rec"])''')
rep('''QE.paired(res["rec"], {k: v[:len(frames)] for k, v in b0["rec"].items()}, frames)''', '''QE.paired(res["rec"], pick(b0["rec"]), frames)''')
rep('''            ctl_hi, ctl_rule = (read_ctl(x, kw["ident"], a.seed, ds.K, out_dir or CTL, n_max) for x in ("hi", "rule"))''', '''            ctl_hi, ctl_rule = (read_ctl(x, kw["ident"], a.seed, ds.K, out_dir or CTL, n_max) if a.every == 1 else None for x in ("hi", "rule"))        # the controls are scored on the first N / all frames: not on a subsample''')
open(p, "w", encoding="utf-8").write(s)
