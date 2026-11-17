import os
p = os.path.join(os.environ["BFLC_ROOT"], "results", "E2E_Champion_Report", "parts", "05_quant.md")
t = open(p, encoding="utf-8").read()
a = t.index("Look at the token head S1 (left of the line)")
b = t.index("The axis starts at 0.5.")
t = t[:a] + "Look at the token head at the decode of record (left of the line) against the same head at a decode chosen on the TRAIN sample and against the champion, and at the earlier systems, which keep five hard steps. " + t[b:]
open(p, "w", encoding="utf-8").write(t)
print("ok")
