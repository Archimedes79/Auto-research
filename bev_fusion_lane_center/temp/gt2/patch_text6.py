p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


# status rows: E305, E309, E310, E311
sub('["E308 / E309", "node tokens of B at the keypoints; the delivery threshold of the connectors", "E308 TIE (pair ranking AUC +0.01, precision at equal recall +0.02 .. +0.06). E309: a lower threshold (0.1) gives lanes found +0.06 .. +0.07 and junction recall +0.15 .. +0.23 with topo F1 flat; confirmation on other networks running", "<span class=\'pill info\'>TIE / running</span>"],',
    '["E308", "node tokens of B at the keypoints", "TIE: the pair ranking improves a little (AUC +0.01), nothing else", "<span class=\'pill info\'>TIE, kept</span>"],\n'
    '    ["E305", "maps head trained together with the pair head", "TIE by the card (topo F1 +0.001 over p2); more pairs at the same precision (recall +0.05), keypoint F1 -0.03; the best network on the snapped read-out", "<span class=\'pill info\'>TIE, kept</span>"],\n'
    '    ["E309", "the delivery threshold of the connectors (0.5 today)", "a lower threshold gives lanes found +0.06 .. +0.07 and junction recall +0.15 .. +0.23 with topo F1 flat when the connectors are drawn from the keypoints", "<span class=\'pill info\'>exploratory</span>"],\n'
    '    ["E310", "do the numbers express what we see? maneuver coverage", "plain graph 52.5 %, connectors p >= 0.1 61.6 %, perfect pairs 64.3 %, perfect scales 67.1 %; the keypoint ends are 1.44 m off", "<span class=\'pill ok\'>measured</span>"],\n'
    '    ["E311", "snap the connector ends to the delivered lane pieces", "<b>KEEP at p >= 0.2 and p >= 0.1 on E287 p2 and E305, both seeds: topo F1 +0.023 .. +0.030 over the plain graph (the written rule asked +0.02), lanes found +0.06 .. +0.08, junction F1 +0.02 .. +0.06, duplicates -0.03 .. -0.05</b>", "<span class=\'pill ok\'>KEEP</span>"],')
sub('"code written and tested; running today (day chain 3, both seeds, scored right after each run)", "<span class=\'pill info\'>running</span>"', '"E303 training (handle scales); E298 and E304 queued behind it (day chain 4)", "<span class=\'pill info\'>running</span>"')
sub('"code written and tested on real frames; E305 (maps and pair head trained together) and E302 run first today because the same-ends comparison points at the pair decision", "<span class=\'pill info\'>running</span>"', '"E306 exact targets and E302 capacity queued (day chain 4, after E303 / E307 / E304)", "<span class=\'pill info\'>queued</span>"')
# lead
sub("a TIE is kept and refined but never worse (D82), VAL only, TEST never touched.\")", "a TIE is kept and refined but never worse (D82), VAL only, TEST never touched. "
    "The result of the day: the delivered graph gets the connectors of the pair head with their ends snapped to the delivered lane pieces, at a low probability threshold: topo F1 +0.03 over the plain graph (the pair head alone, drawn from the keypoints at 0.5: +0.005), lanes found +0.07, junction recall +0.09 .. +0.16, duplicates -0.05 (E311).\")")
open(p, "w", encoding="utf-8").write(s)
print("ok")
