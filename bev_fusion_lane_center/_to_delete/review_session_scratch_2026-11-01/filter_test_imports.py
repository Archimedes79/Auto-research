p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
old = '"test_imports": sorted(info["test_imports"]),'
new = '"test_imports": sorted(x for x in info["test_imports"] if x.split(" -> ")[1] not in files),'
assert s.count(old) == 1
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
