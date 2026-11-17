import re
p = "evaluation/jdup.py"
s = open(p, encoding="utf-8").read()
a = "    from qends_eval import COVER_POS, add_connectors, make_pairs\n"
assert s.count(a) == 1
s = s.replace(a, "    from jwire import kept_ends\n    from qends_eval import cover_pos, add_connectors, make_pairs\n")
i0 = s.index("            kept = [kd for i, kd in enumerate(bk[\"kind\"]) if not (kd == \"link\"")
i1 = s.index("            assert len(kept) == len(raw[\"pts\"]) - n_c")
s = s[:i0] + "            kept = [bk[\"kind\"][i] for i in kept_ends(graph.lines, bk[\"kind\"], bk[\"piece_links\"], lm, pairs, cover_pos())[0]]                       # the kinds of the pieces the cover drop keeps (E325: one place for it)\n" + s[i1:]
assert "COVER_POS" not in s, [m.start() for m in re.finditer("COVER_POS", s)]
open(p + ".new_s292", "w", encoding="utf-8", newline="\n").write(s)
print("ok")
