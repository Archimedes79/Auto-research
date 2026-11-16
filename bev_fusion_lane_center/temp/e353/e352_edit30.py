p = "temp/e353/build_notes.md"
s = open(p, encoding="utf-8").read()
s = s.replace(" The effect on the cards is inside the bound above (the read-out of B's delivered graph reads the pieces, not the section ids alone).", " The effect on the cards is inside the bound above (the null gate is the test of it).")
s = s.replace("; the 8-frame number of 0.285 m was the same two keypoints.", ".")
open(p, "w", encoding="utf-8").write(s)
