import re
p = "tests/test_student_chain.py"
s = open(p, encoding="utf-8").read()
old = "     global clip scales them down\nRun from"
new = ("     global clip scales them down\n"
       "  J  the trainable student recomputes its backbone in the backward (the memory fix of the smoke run: 9 frames x backbone activations were 16.3 GB): the option follows\n"
       "     student_trainable; the encoder runs twice with it and once without; gradients of every student tensor equal (relative 1e-5), also through the chain; BatchNorm\n"
       "     statistics unmoved; nothing recomputed in evaluation mode, under no_grad, or for a frozen student\n"
       "Run from")
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
