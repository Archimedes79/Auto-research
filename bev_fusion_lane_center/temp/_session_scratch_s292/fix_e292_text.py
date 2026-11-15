p = "experiments/E292_matrix-nms-layer/experiment.md"
s = open(p, encoding="utf-8").read()
for a, b in (("even with the hard step on top (ah) the system stays 0.024 topo F1 below c2.", "even with the hard step on top (ah) the system stays 0.025 / 0.022 topo F1 below c2."),
             ("with junction F1 -0.05 / -0.03 and vertices +2 against the hard step (graph-only); arm a pays more (junction -0.06 graph-only, vertices +4)", "with junction F1 -0.048 / -0.031 and vertices +2.1 / +0.8 against the hard step (graph-only); arm a pays more (junction -0.073 / -0.059 against c2 graph-only, vertices +4.2 / +4.3)")):
    assert s.count(a) == 1, a[:50]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
