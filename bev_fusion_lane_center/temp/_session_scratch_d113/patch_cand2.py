import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_next_candidates.md")
s = open(p, encoding="utf-8").read()
a = "champion long links: wrong pairing 34 - 36 %, false end 33 - 36 %, tp 17 - 18 % |"
assert s.count(a) == 1
s = s.replace(a, "champion long links: wrong pairing 34 - 36 %, false end 33 - 36 %, tp 17 - 18 %; CPU chord probe (item 4): the written AUC gate NOT met (3 of 5 folds), AP over geometry + tok +0.034 .. +0.143 on 5 of 5 folds: an amended gate is the lead's call |")
open(p, "w", encoding="utf-8").write(s)
print("ok")
