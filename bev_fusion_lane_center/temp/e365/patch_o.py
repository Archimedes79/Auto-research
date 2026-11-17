import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("# the part of the loss from the sources with a GT successor does not move with W: the gradient of a logit of a G-source is the same up to the shared softmax normalisation only through its own row", "# the sources with a GT successor are not touched by W: the gradient of the row of frame 0, source 0 (G non-empty) is identical, the none logit of source 1 (G empty) moves")
rep('''abs(row["total"] - 2.8015905618667603) < 1e-9 and abs(row["nll_s"] - 1.288527250289917) < 1e-9 and abs(tot / 8250.530359720806 - 1) < 1e-12''', '''abs(row["total"] - 2.8015905141830446) < 1e-12 and abs(row["nll_s"] - 1.288527250289917) < 1e-12 and abs(tot / 8250.530362168049 - 1) < 1e-14''')
rep("recorded before the change: seed 0, bs 2, K 48)", "recorded from the committed code before the changes, 2 threads: seed 0, bs 2, K 48)")
rep("reproduce the recorded loss 2.80159 and the weights)", "reproduce the recorded loss 2.8015905 and the weights, also with --none-weight 1)")
rep('''    dflt = JKT.parse_args(["--name", "x"])''', '''    dflt = JKT.parse_args(["--name", "x", "--none-weight", "1.0"])''')
open(p, "w", encoding="utf-8").write(s)
