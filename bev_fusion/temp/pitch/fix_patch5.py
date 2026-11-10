p = "temp/pitch/patch_bev_eval5.py"
s = open(p, encoding="utf-8").read()
a = s.index("    ('''    ident = rs.manifest")
b = s.index("])\nprint")
s = s[:a] + s[b:]
open(p, "w", encoding="utf-8").write(s)
print("ok")
