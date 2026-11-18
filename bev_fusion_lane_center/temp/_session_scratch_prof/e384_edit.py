import sys
row = ("| a recursive node state in the node head of B' (the previous keyframe's B' output moved with the ego pose, one frame back; step 0 on 851 VAL frames, no training) | E384 (planned under D118, closed at step 0) | "
       "oracle coverage ceiling of the previous keyframe's output net of the identity null +0.0021 .. +0.0037 coverage points (bar +0.006); pose-minus-identity AP lift +0.0017 / +0.0018 (bar +0.005); 1 true to 8 false carried-over tokens | "
       "CUT by hand: critic step 0 (the bars were written after the numbers were seen); no code, no GPU hour; the card stays ready to run, `closed_questions.md` holds the row |")
for f in sys.argv[1:]:
    s = open(f, encoding="utf-8").read()
    a = "| TIE, the champion takes no past block |"
    assert s.count(a) == 1, (f, s.count(a))
    s = s.replace(a, a + "\n" + row)
    b = "Not run before S4 has a verdict."
    assert s.count(b) == 1, f
    s = s.replace(b, b + " **Status after the night of 2026-11-18:** the first card of this step (E384, the state in the node head, D118) was measured at step 0 and closed by the ceiling (table 9.1, last row): the previous keyframe's B' output adds at most +0.002 to +0.004 coverage points, so a feature-level state is re-opened only with a new mechanism; the E2E part that was trained is S4 (E372).")
    c = "A recursive state was tried at three places of the lane system."
    assert s.count(c) == 1, f
    s = s.replace(c, "A recursive state was tried at three places of the lane system and probed at a fourth (step 0 only).")
    open(f, "w", encoding="utf-8", newline="\n").write(s)
