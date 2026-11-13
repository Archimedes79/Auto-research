p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('"training (both seeds), result expected ~13:40", "<span class=\'pill info\'>running</span>"', '"<b>KEEP, both seeds: topo F1 over the plain graph +0.057 / +0.062 (E305: +0.030 / +0.029), lanes found +0.11 / +0.13, junction F1 +0.05 / +0.07, duplicates -0.06 / -0.05; entry / exit recall 0.75 / 0.77 (E305 0.65 / 0.64), pair recall 0.30 / 0.33 (E305 0.21 / 0.22)</b>", "<span class=\'pill ok\'>KEEP</span>"')
sub("The result of the day: the delivered graph gets the connectors of the pair head with their ends snapped to the delivered lane pieces, at a low probability threshold: topo F1 +0.03 over the plain graph (the pair head alone, drawn from the keypoints at 0.5: +0.005), lanes found +0.07, junction recall +0.09 .. +0.16, duplicates -0.05 (E311).",
    "The result of the day: three changes together, the narrower end heat of the junction maps (E307), the pair head trained with the maps and with learned handle scales (E305, E303) and connectors whose ends are snapped to the delivered lane pieces at a low probability threshold (E311): the delivered graph gains +0.057 / +0.062 topo F1 over the graph without connectors (the pair head alone, connectors drawn from the keypoints at 0.5: +0.006 / +0.004), lanes found +0.11 / +0.13, junction F1 +0.05 / +0.07, duplicates -0.06 / -0.05 (E312).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
