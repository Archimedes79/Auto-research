p = "training/grid/README.md"
s = open(p, encoding="utf-8", newline="").read()
a = "the 1 m maps (`fields`). Test: `python jgt.py`. |"
assert s.count(a) == 1
b = "the 1 m maps (`fields`). `_groups` (the single-linkage groups of the keypoints of the pair head, called per frame by `JEnds.rank_matrix` and by the loader) is vectorised since E342 (all pairs at once, components by repeated squaring of the adjacency matrix, labels by first member; 0.80 -> 0.095 ms per set of 13 ends; the python loop stays in the selftest as the oracle and is compared on random clustered sets, the edge cases (boundary `<=`, a zero heading, float32 headings with self-dot above 1, a shuffled path of 32 ends) and chains). Test: `python jgt.py`. |"
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
