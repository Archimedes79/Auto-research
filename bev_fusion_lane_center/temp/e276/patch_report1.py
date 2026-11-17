p = "temp/e276/e276_build_report.py"
L = open(p, encoding="utf-8").read().split("\n")
a = next(i for i, l in enumerate(L) if l.startswith('parts.append("<h2>1. E275'))
b = next(i for i, l in enumerate(L) if l.startswith("e275fig ="))
new = '''p6txt = ("E278 (no card, scoring only): the model before the joint refinement phase has topology F1 " + m2(P6, "topo_F1") + ", long-link precision " + m2(P6, "link_long_P", "nodes") + " and duplicates " + m2(P6, "dup_frac") +
         " against " + m2(B, "topo_F1") + ", " + m2(B, "link_long_P", "nodes") + " and " + m2(B, "dup_frac") + " after it, but junction F1 " + m2(P6, "junction_F1") + " against " + m2(B, "junction_F1") + " and " + m2(P6, "vert100_dp", fmt="{:.1f}") +
         " vertices per 100 m against " + m2(B, "vert100_dp", fmt="{:.1f}") + ": the joint phase helps the links and the topology and costs junction F1 and vertices.") if all(P6) else "<span class='pending'>E278 (the model before the joint phase) is not fully scored yet.</span>"
parts.append("<h2>1. E275: the close-node step (decided: KEEP)</h2>" + table(["", "baseline B", "B + step", "B before the joint phase (E278)", "that + step"], rows) +
             "<p>The step thins the dense node row (about 9 % of the nodes). The scorer counts 65 % of the dropped nodes as true, and the pictures show a node row thinned along the lanes: it is mainly a density cut that also cleans the junction links, not a removal of false nodes only.</p><p>" + p6txt + "</p>")'''.split("\n")
L[a:b] = new
open(p, "w", encoding="utf-8").write("\n".join(L))
print("patched", a, b)
