p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace("def gate_md(rows, only_worst=3):", "def gate_md(rows):")
s = s.replace('''    for seed, d in data.items():
        n = d["null"]
        out += [f"### seed {seed}, {key}''', '''    for seed, d in data.items():
        n = d["null"]
        if None in n:
            out += [f"### seed {seed}: the null cards are missing", ""]
            continue
        out += [f"### seed {seed}, {key}''')
open(p, "w", encoding="utf-8").write(s)
