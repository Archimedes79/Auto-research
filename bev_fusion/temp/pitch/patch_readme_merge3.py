p = "evalsuite/README.md"
s = open(p, encoding="utf-8").read()
new = open("../bev_fusion/temp/pitch/readme_section3.md", encoding="utf-8").read().rstrip("\n") + "\n"
a = s.index("## Try it")                                             # the engine's blocks: Try it ... From Python
b = s.index("## The class interface")
c = s.index("## Where the comparison happens")
engine = s[a:b].rstrip("\n") + "\n"
engine = engine.replace("## Try it", "### Try it").replace("## The three boxes", "### The three boxes").replace("## What you get", "### What you get")
rest = s[c:].replace("## Where the comparison happens", "### Where the comparison happens").replace("## Port it to your network", "### Port it to your network")
rest_more_at = rest.index("## More")
rest = rest[:rest_more_at] + "## More" + rest[rest_more_at + len("## More"):]
out = new + "\n" + engine + "\n" + rest
open(p, "w", encoding="utf-8").write(out)
print(len(s), "->", len(out))
