"""One-off: repair the whitespace-skipping line of _stream_records in training/cache_bevfusion_features.py (a heredoc had turned its escapes into raw control characters)."""
p = "training/cache_bevfusion_features.py"
lines = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(lines) if l.strip().startswith("while i < len(buf) and buf[i] in"))
j = i
while not lines[j].rstrip().endswith(":") or lines[j].rstrip().endswith('" '):
    j += 1
    if lines[j].rstrip().endswith('",:') or lines[j].rstrip().endswith(",\":"):
        break
fixed = "            while i < len(buf) and buf[i] in \" \\n\\r\\t,\":"
end = next(k for k in range(i, i + 4) if lines[k].rstrip().endswith('":'))
lines[i:end + 1] = [fixed]
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("fixed lines", i + 1, "to", end + 1, "->", fixed)
