import io
p = "experiments/E319_heading-guided-exchange/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep("the spread of E307 there is 0.0013: the fall is larger than that spread on seed 1 and about three times on seed 0)", "the spread of E307 there is 0.0013: the falls are about 3 and 6 times that spread)")
rep("Cost of the arm: about 1.8 GPU hours (the two maps stages), the pair stages (about 4 GPU hours each) were not spent.", "Cost of the arm: about 1.8 GPU hours (the two maps stages, 3391 s + 3261 s); the pair stages were not spent.")
io.open(p, "w", encoding="utf-8", newline="").write(s)
