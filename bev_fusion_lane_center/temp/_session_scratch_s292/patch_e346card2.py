import io
p = "experiments/E346_one-graph-code-ceiling/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
for a in ("g0", "g1", "l2", "l1"):
    old = f', "names": ["e346_{a}"]}}'
    assert s.count(old) == 1, old
    s = s.replace(old, "}")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("reverted")
