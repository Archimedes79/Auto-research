p = "experiments/E344_e2e-completion-plan/experiment.md"
s = open(p, encoding="utf-8").read()
a = "topo F1 of the decoded GT at least B's own representation oracle (`qformat.oracle_prediction`, same frames, same scorer; E273 l.284: 0.97) minus 0.005"
assert s.count(a) == 1
b = "topo F1 of the decoded GT at least B's own representation oracle (`qformat.oracle_prediction`, same frames, same scorer; E273 l.284: 0.97; a hand check of 31 VAL frames by the E346 critics: the old GT against itself 1.0000, the undecoded lane GT 0.9854, B's code at 2 m 0.9684, so the code itself loses about 0.017, not 0.03) minus 0.005"
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
