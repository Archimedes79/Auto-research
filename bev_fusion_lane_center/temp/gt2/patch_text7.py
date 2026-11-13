p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('["E289", "the points are unstable"', '["E307", "narrower end heat targets (sigma 1.0 m instead of 1.5 m)", "<b>KEEP, both seeds: keypoint F1 0.723 / 0.705 against 0.630 / 0.642, crowded entry recall 0.79 against 0.54 / 0.58, entry recall beyond 35 m +0.09 / +0.10</b>", "<span class=\'pill ok\'>KEEP</span>"],\n    ["E303", "two learned handle scales per pair (the shape of the connector)", "TIE on the written rule (connectors from the keypoints); on the snapped read-out the scales add +0.006 / +0.011 topo F1 and give the best numbers of the thread so far: +0.032 / +0.036 over the plain graph", "<span class=\'pill info\'>TIE, used</span>"],\n    ["E312", "the pair head (maps + pairs together, handle scales) on the E307 maps, snapped read-out", "training (both seeds), result expected ~13:40", "<span class=\'pill info\'>running</span>"],\n    ["E289", "the points are unstable"')
sub('"E303 training (handle scales); E298 and E304 queued behind it (day chain 4)", "<span class=\'pill info\'>running</span>"', '"E298 and E304: not run (E304 stopped at step 200 when E307 made the single-factor test moot; E298: the pair decision is at the level of a perfect classifier)", "<span class=\'pill no\'>dropped</span>"')
sub('"E306 exact targets and E302 capacity queued (day chain 4, after E303 / E307 / E304)", "<span class=\'pill info\'>queued</span>"', '"E306 exact targets and E302 capacity: cards kept, not run (the slots went to the ends)", "<span class=\'pill info\'>not run</span>"')
open(p, "w", encoding="utf-8").write(s)
print("ok")
