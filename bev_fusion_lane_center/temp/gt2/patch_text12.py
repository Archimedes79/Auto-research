p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('"training (both seeds), result expected ~15:45", "<span class=\'pill info\'>running</span>"', '"<b>TIE with E312</b> (topo F1 +0.0593 / +0.0597 against +0.0566 / +0.0615 over the plain graph; lanes found +0.012 / +0.006 more, entry / exit recall 0.73 / 0.73 against 0.75 / 0.77): E312 stays the final system", "<span class=\'pill info\'>TIE, kept</span>"')
sub("The first three changes are cheap read-outs, the last is a retraining of the junction maps; E313 (sigma 0.7 m) asks whether it goes on.", "The first three changes are cheap read-outs, the last is a retraining of the junction maps; E313 / E314 (sigma 0.7 m) asked whether it goes on: the maps get a little better, the delivered graph does not (TIE), so E312 is the final system. On the same pairs the learned handle scales are worth +0.012 / +0.013 topo F1 (row with chord curves).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
