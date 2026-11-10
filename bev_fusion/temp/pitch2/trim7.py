p = "../bev_fusion/temp/pitch2/patch_round7.py"
s = open(p, encoding="utf-8").read()
a = s.index("def patch")
b = s.index("# ----", s.index("# ------------------------------------------------------------------------------------------------ api.py: no fallback") - 1)
head = s[a:s.index("# ----")]
rest = s[s.index("# ------------------------------------------------------------------------------------------------ api.py: no fallback"):]
open("../bev_fusion/temp/pitch2/patch_round7b.py", "w", encoding="utf-8").write(head + rest)
print("ok")
