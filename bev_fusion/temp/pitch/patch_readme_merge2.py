p = "evalsuite/README.md"
s = open(p, encoding="utf-8").read()
new = open("../bev_fusion/temp/pitch/readme_section2.md", encoding="utf-8").read().rstrip("\n") + "\n"
a = s.index("## The class interface: `Evaluation` and `Runner`")
b = s.index("## Where the comparison happens")
out = s[:a] + new + "\n" + s[b:]
open(p, "w", encoding="utf-8").write(out)
print(len(s), "->", len(out))
