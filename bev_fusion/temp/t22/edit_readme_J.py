"""scratch: the tests/README.md row of test_student_chain gets section J"""
p = "tests/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
idx = [i for i, ln in enumerate(lines) if ln.startswith("| `test_student_chain.py`")]
assert len(idx) == 1, idx
ln = lines[idx[0]]
print(ln[-400:])
assert ln.rstrip().endswith("|")
add = (" J: the memory fix of the smoke run (9 frames x encoder activations + lift frustum features were 16.3 GB on a 12 GB card): the trainable student's encoder and the lift in `camera_bevs` are recomputed in the backward; "
       "independent oracle (the camera_bevs before the student, an encoder without recomputation) on 3 samples with 3 cameras equal bit for bit in outputs, every gradient and every state tensor; call counts after a backward; "
       "the BatchNorm guard; the bytes saved for the backward fall (8 planted bugs, 8 killed).")
lines[idx[0]] = ln.rstrip()[:-1].rstrip() + add + " |"
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
