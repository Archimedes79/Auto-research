p = "reports/build/reports/build_status21_report.py"
s = open(p, encoding="utf-8").read()
R = [
 ('"Q about 0.74 at about 10 vertices per 100 m; equal to the 628 ms control within the noise band; plateau since loop 18"', '"Q about 0.74 at about 10 node ids (about 12 on the budget basis) per 100 m; equal to the 628 ms control within the noise band; plateau since loop 18"'),
 ('Loop 21 improved form (sparser, simpler, verified) and not substance.', 'Loop 21 improved form (a smaller delivered graph, a simpler verified assembly) and not substance.'),
 ('Douglas-Peucker with 0.2 m on every polyline; ends and shared nodes are kept, so topology is unchanged and no vertex moves more than 0.2 m.', 'Douglas-Peucker with 0.2 m on every polyline; ends and shared nodes are kept, so topology is unchanged and no vertex moves more than 0.2 m. It drops redundant nodes on straight runs: the geometry is the same, the node count is lower.'),
]
for a, b in R:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
