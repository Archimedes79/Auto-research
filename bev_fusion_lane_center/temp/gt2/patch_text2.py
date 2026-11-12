p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub('"see the maps section: kp_f1 against B\'s event heat 0.40", "<span class=\'pill ok\'>trained</span>"',
    '"KEEP on all three context arms, two seeds: entry / exit keypoint F1 0.64 against B\'s event heat 0.39 (rule +0.05); base arm j1 (dilated context). The joint arm (pyramid trained together) ran at 11 s per step and was stopped at step 1000, not judged", "<span class=\'pill ok\'>KEEP</span>"')
sub('"see the pairs section", "<span class=\'pill info\'>night run</span>"',
    '"TIE on all three arms (topo F1 +0.00 .. +0.006 over the plain graph, the rule asked +0.02); lanes found +0.02, junction recall +0.04 .. +0.11, duplicates -0.02 .. -0.03; the order features help (p1 over p0), the evidence along the curve does not add", "<span class=\'pill info\'>TIE, kept</span>"')
sub('["E288", "the close-node step', '["E308 / E309", "node tokens of B at the keypoints; the delivery threshold of the connectors", "E308 TIE (pair ranking AUC +0.01, precision at equal recall +0.02 .. +0.06). E309: a lower threshold (0.1) gives lanes found +0.06 .. +0.07 and junction recall +0.15 .. +0.23 with topo F1 flat; confirmation on other networks running", "<span class=\'pill info\'>TIE / running</span>"],\n    ["E288", "the close-node step')
sub('"code written and tested, queued behind the night chain (day chain 1)", "<span class=\'pill info\'>queued</span>"', '"code written and tested; running today (day chain 3, both seeds, scored right after each run)", "<span class=\'pill info\'>running</span>"')
sub('"code written and tested on real frames (CPU), queued (day chains 2 and 3)", "<span class=\'pill info\'>queued</span>"', '"code written and tested on real frames; E305 (maps and pair head trained together) and E302 run first today because the same-ends comparison points at the pair decision", "<span class=\'pill info\'>running</span>"')
sub("Three context depths (none, four dilated residual blocks, dilated blocks plus one attention layer at 4 m) and one joint arm (the pyramid trained together with the head) were trained, two seeds each, 3000 steps.",
    "Three context depths (none, four dilated residual blocks, dilated blocks plus one attention layer at 4 m) were trained, two seeds each, 3000 steps; a joint arm (the pyramid trained together with the head) was started and stopped at step 1000 because it ran at 11 s per step (the frozen arms: 0.84 s), it is not judged.")
sub("q2 is p2 on the maps of the joint arm. The rule control", "E308 is p2 with the node tokens of B at the keypoints; q2 (p2 on the joint maps) did not run because the joint arm was stopped. The rule control")

# the reading of the pairs: replace PAIRS2 and add PAIRS3 / PAIRS4
i = s.index("PAIRS2 = (")
j = s.index("PRUNE = (")
pairs2 = '''PAIRS2 = ("<div class='note'>Reading of the table above (VAL 873 frames, two seeds, the plain delivered graph of B + the hard close-node step as the control): <b>all three arms are a TIE by the written rule</b> (topo F1 +0.00 .. +0.006, the rule asked +0.02 on both seeds), but the connectors do what they are for: lanes found +0.017 .. +0.027, junction recall +0.04 .. +0.11, duplicates -0.02 .. -0.03, F1@1.0 up, no guard broken. The order features help (p1 over p0: +0.004 / +0.003 topo F1, +0.04 pair recall at the same precision), the evidence along the Hermite curve does not add (p2 against p1 inside the band). The rule control (strict one-to-one matching, the same keypoints as the E284 head) delivers about three times as many pairs and is wrong for two thirds of them (precision 0.33 against 0.74), with a ceiling of 0.43 - 0.53 (both ends of a pair found). The pair flip rate between keyframes is 20 % (E290).</div>")

PAIRS3 = ("<div class='note'><b>The weak spot is the pair decision, not the ends.</b> Counted against the E284 head on its own keypoints the new heads look much worse (pair precision at the recall of E284: 0.52 - 0.60 against 0.74), but that comparison is not like for like: the new keypoints find more, harder ends. The table restricts the candidates to GT ends that ALL three systems found (3579 / 3476 GT pairs, about 11 000 candidates each): on the same ends the E284 head ranks the pairs clearly better (AUC 0.975 / 0.976 against 0.94 / 0.95) and says yes about twice as often at the same precision (recall 0.67 / 0.71 against 0.34 / 0.38 at precision 0.94). It is not the geometry: the heading of the new keypoints is better (4.4 degrees median error 3.2 against 8.9 / 7.4), the position slightly worse (1.38 m against 1.18 m), and the pair part of both heads has the same size (160 k parameters). What the E284 head has that the new one does not: its heat, offset and heading are trained TOGETHER with the pair decision (E305 asks for exactly that), and its tokens come from B's node head (E308 adds them: AUC +0.01, a part of the gap). E302 (degree budget with a dustbin) goes after the second symptom, the unwillingness to deliver.</div>")

PAIRS4 = ("<div class='note'><b>The delivery threshold (E309).</b> The connectors are delivered at p >= 0.5 today. Delivered lower, the graph gets more lanes and more junction recall for free (no training): at 0.1 on the E308 pairs lanes found +0.056 / +0.070 and junction recall +0.15 / +0.23 over the plain graph, duplicates -0.06, F1@1.0 +0.014 / +0.025; topo F1 stays flat (+0.005 .. +0.007) because the extra curves cost on it what they gain, junction F1 moves between -0.014 and +0.014. This sweep was run on the E308 dumps first (exploratory, on VAL); the card E309 fixes the rule and the independent confirmation on E287 p2 and on the networks of today. It is a VAL-selected setting, TEST untouched.</div>")

'''
s = s[:i] + pairs2 + "\n" + pairs3_placeholder if False else s[:i] + pairs2 + "\n" + s[j:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
