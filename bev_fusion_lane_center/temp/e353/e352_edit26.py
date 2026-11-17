p = "temp/e353/e352_mutants.py"
s = open(p, encoding="utf-8").read()
a = s.index('("group_means_misaligned"')
b = s.index('["groups_variant_references"]),', a) + len('["groups_variant_references"]),')
seg = s[a:b]
seg2 = seg.replace("\n", chr(92) + "n")
open(p, "w", encoding="utf-8").write(s[:a] + seg2 + s[b:])
