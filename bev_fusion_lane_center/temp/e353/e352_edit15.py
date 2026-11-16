p = "temp/e353/e352_mutants.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    ("ranking_reversed", "    return \\"\\".join", "    return \\"\\".join", ["points_ladder_counts_and_reference_channels"]),''', '''    ("group_ranking_reversed", "torch.argsort(calib.s[tap].double().view(G, gs).sum(1), stable=True)", "torch.argsort(calib.s[tap].double().view(G, gs).sum(1), descending=True, stable=True)", ["points_ladder_counts_and_reference_channels"]),''')
open(p, "w", encoding="utf-8").write(s)
