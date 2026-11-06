"""scratch: the docstring lines of section J of tests/test_student_chain.py"""
p = "tests/test_student_chain.py"
s = open(p, encoding="utf-8").read()
a = s.index("  J  the trainable student recomputes")
b = s.index("Run from the bev_fusion folder: wsl")
new = ("  J  the trainable student recomputes its backbone and the lift in the backward (the memory fix of the smoke run: 9 frames x backbone activations + frustum features were 16.3 GB\n"
       "     on a 12 GB card). Independent oracle (the camera_bevs of the commit before the student on a batch with the student's frame 0, no recomputation) on 3 samples with 3\n"
       "     cameras: outputs, every gradient and every state tensor / buffer equal bit for bit; encoder and lift call counts after a backward (training 2 x, evaluation mode 1 x,\n"
       "     the encoder's switch alone); no recomputation of an encoder whose BatchNorm is not held in eval mode; the bytes autograd saves for the backward fall (encoder to a\n"
       "     quarter at most, camera path to half at most)\n")
open(p, "w", encoding="utf-8").write(s[:a] + new + s[b:])
print("ok")
