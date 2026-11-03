p = "autoresearch/evalsuite/comparison/mutation_check.py"
s = open(p, encoding="utf-8").read()
a = s.index('    ("maps.py", "astype(np.float32)')
b = s.index('    ("maps.py", "[(p & near_g & v).sum()')
s = s[:a] + '    ("maps.py", "np.stack(cols, 1).astype(np.float32)", "np.stack(cols, 1).astype(np.float16)", "tolerant_counts_many: counts in half precision"),\n' + s[b:]
open(p, "w", encoding="utf-8").write(s)
p = "autoresearch/evalsuite/comparison/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
out, seen = [], False
for ln in lines:
    if ln.startswith("- `tolerant_counts_many("):
        if seen:
            continue
        seen = True
    out.append(ln)
open(p, "w", encoding="utf-8").write("\n".join(out))
