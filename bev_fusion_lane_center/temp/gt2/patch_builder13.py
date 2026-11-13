p = "temp/jr/build_junction_report.py"
s = open(p, encoding="utf-8").read()
old = '''             ("<b>E312: + the narrower end heat E307 (heat sigma 1.0 m), scales, snapped, p >= 0.1</b>", "runs/e312_p_s{s}/ends_card_snap_thr0.1.json")]'''
assert old in s
new = '''             ("E312 with chord curves instead of the learned scales (the same pairs)", "runs/e312_p_s{s}/ends_card_snap_chord_thr0.1.json"),
             ("<b>E312: + the narrower end heat E307 (heat sigma 1.0 m), scales, snapped, p >= 0.1 (the final system)</b>", "runs/e312_p_s{s}/ends_card_snap_thr0.1.json"),
             ("E314: the same on the sigma 0.7 m maps of E313 (TIE with E312)", "runs/e314_p_s{s}/ends_card_snap_thr0.1.json")]'''
s = s.replace(old, new, 1)
s = s.replace("(\"E287 p2: pair head, connectors from the keypoints, p >= 0.5\", \"runs/e287_p2_s{s}/ends_card.json\"),", "(\"E287 p2: pair head, connectors from the keypoints, p >= 0.5\", \"runs/e287_p2_s{s}/ends_card.json\"),", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
