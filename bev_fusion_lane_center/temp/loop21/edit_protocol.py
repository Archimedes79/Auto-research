p = "evaluation/PROTOCOL.md"
t = open(p, encoding="utf-8").read()
key = "- Judge new work by **Q** at vertices per 100 m <= 14.3"
i = t.index(key)
j = t.index("\n", i)
add = "\n- Vertex count (D20): the budget is quoted for the graph after the output stage `simplify_dp` (0.2 m, registry names `*_dp`, `assemble(dp_eps=0.2)`); tables also give the node-id count of the unsimplified graph; a Q comparison names which count it uses. The stage is off by default."
open(p, "w", encoding="utf-8").write(t[:j] + add + t[j:])
