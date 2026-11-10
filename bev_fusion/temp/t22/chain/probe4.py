import sys, os
src = open("tests/test_student_chain.py").read()
cut = src.index("check(torch.equal(seen1[0], other[:1])")
__file__ = os.path.abspath("tests/test_student_chain.py")
exec(compile(src[:cut], "tests/test_student_chain.py", "exec"))
print("seen1 == other", torch.equal(seen1[0], other[:1]), "max |c1-c0|", float((c1 - c0).abs().max()), "max c0", float(c0.abs().max()), "max |p0-p1|", float((echo["prob16"] - other).abs().max()))
