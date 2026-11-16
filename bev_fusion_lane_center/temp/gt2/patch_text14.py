p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('"E298 and E304: not run (E304 stopped at step 200 when E307 made the single-factor test moot; E298: the pair decision is at the level of a perfect classifier)", "<span class=\'pill no\'>dropped</span>"',
    '"E298 (driven pairs weighted 2, on the E312 system): <b>KEEP, both seeds: the recall of the pairs the ego drives 0.495 / 0.511 against 0.373 / 0.398 (+0.12 / +0.11)</b>, the other pairs keep their recall, the snapped topo F1 stays at E312\'s level (-0.006 / +0.000); E304 (far-range weight) was stopped at step 200 when E307 made the single-factor test moot", "<span class=\'pill ok\'>E298 KEEP</span>"')
open(p, "w", encoding="utf-8").write(s)
print("ok")
