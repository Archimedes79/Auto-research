p = "evaluation/tests/test_jprune_b.py"
s = open(p, encoding="utf-8").read()
old_a = s.index("        where = -np.ones(int(max(d[\"idx\"].max(), d[\"section\"].max())) + 1, int)")
old_b = s.index("        d[\"idx\"] = d[\"idx\"][o]")
s = s[:old_a] + "        d[\"section\"] = d[\"section\"][o]                                                                   # section starts are indices of the unfiltered list, `idx` maps them to the kept nodes\n" + s[old_b:]
s = s.replace("    for d in perm:                                                                                       # the same graph with its nodes in another order (links and section starts renumbered through the filtered list)", "    for d in perm:                                                                                       # the same graph with its nodes in another order (the links renumbered)")
open(p, "w", encoding="utf-8").write(s)
