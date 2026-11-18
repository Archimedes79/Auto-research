import io
p = "experiments/E346_one-graph-code-ceiling/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
for a in ("g0", "g1", "l2", "l1"):
    old = f'"run_ids": ["E346.{a}"]}}'
    assert s.count(old) == 1, old
    s = s.replace(old, f'"run_ids": ["E346.{a}"], "names": ["e346_{a}"]}}')
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
