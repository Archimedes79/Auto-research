import io
p = "evaluation/README.md"
s = io.open(p, encoding="utf-8").read()
a = "16 mutants of the pure parts killed)."
assert s.count(a) == 1
s = s.replace(a, "17 mutants of the pure parts killed; `check-arrays` and `check-nodes` (CPU) re-run on the real E363 folders with the scratch versions' results; the promoted `dump` was NOT re-run after the move: the GPU window had closed, the E363 dumps were made by the scratch version).")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
