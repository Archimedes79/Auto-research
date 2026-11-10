p = "evalsuite/README.md"
s = open(p, encoding="utf-8").read()
new = open("../bev_fusion/temp/pitch/readme_section.md", encoding="utf-8").read().rstrip("\n") + "\n"
a = s.index("## The class interface: `Evaluation(project)`")
b = s.index("## Where the comparison happens")
c = s.index("## The class interface: the methods")
head = s[:a] + new + "\n"
middle = s[b:c].rstrip("\n") + "\n"
out = head + middle
open(p, "w", encoding="utf-8").write(out)
print(len(s), "->", len(out))
