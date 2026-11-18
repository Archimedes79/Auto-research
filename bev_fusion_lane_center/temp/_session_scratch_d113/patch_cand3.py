import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_next_candidates.md")
s = open(p, encoding="utf-8").read()
a = "E380 step 0 (the CPU chord probe, about 1 h, decides whether the 4.5 GPU h are spent) in parallel because it needs no GPU,"
assert s.count(a) == 1
s = s.replace(a, "E380 step 0 (the CPU chord probe: DONE 20:12, the written gate not met, E380 CLOSED by the lead at 20:40, see item 4),")
open(p, "w", encoding="utf-8").write(s)
print("ok")
