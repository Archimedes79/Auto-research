p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()
old = "the learned handle scales add about +0.002 / +0.007 on top (E303), and the narrower end heat is what lifts the ends"
assert old in s
s = s.replace(old, "the E303 pair head with learned handle scales reaches +0.032 / +0.036 (on the same pairs the scales are worth +0.006 / +0.011 against chord curves), and the narrower end heat is what lifts the ends", 1)
open(p, "w", encoding="utf-8").write(s)
