p = "tests/test_student_chain.py"
s = open(p, encoding="utf-8").read()
old = "     statistics unmoved; nothing recomputed in evaluation mode, under no_grad, or for a frozen student\n"
new = ("     statistics unmoved; nothing recomputed in evaluation mode, under no_grad, or for a frozen student; the same for the lift in `camera_bevs` (runs twice per sample, same\n"
       "     gradients)\n")
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
