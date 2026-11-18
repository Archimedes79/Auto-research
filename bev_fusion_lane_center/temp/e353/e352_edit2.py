p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace("        out += (group_edits if groups else PP.edits_for)(model, u, f, calib, how, rng)\n", "        out += group_edits(model, u, f, calib, how, rng) if groups else PP.edits_for(u, f, calib, how, rng)\n")
open(p, "w", encoding="utf-8").write(s)
