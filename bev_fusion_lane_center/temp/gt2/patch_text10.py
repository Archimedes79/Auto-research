p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()
old = "After (E303 pairs with learned handle scales, connector ends snapped to the delivered pieces, p >= 0.1): 0.41 of the GT pairs of a frame, 40 frames without one, coverage 0.83, precision 0.85."
assert old in s
s = s.replace(old, "After (E312: narrower end heat, pairs with learned handle scales, connector ends snapped to the delivered pieces, p >= 0.1): 0.52 of the GT pairs of a frame, 27 frames without one, coverage 0.85, precision 0.86.", 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
