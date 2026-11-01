p = "training/pipeline/p16_infer.py"
t = open(p, encoding="utf-8").read()
a = "    if dp_eps > 0 and not aux:  # output stage (E257 / D20): drop interior vertices of straight runs, geometry kept within dp_eps; off by default\n        g = simplify_dp(g, dp_eps)\n"
b = "    if dp_eps > 0:  # output stage (E257 / D20): drop interior vertices of straight runs, geometry kept within dp_eps; off by default; aux describes the unsimplified vertices\n        if aux:\n            raise ValueError(\"dp_eps and aux cannot be combined: the token map of aux refers to the unsimplified vertices\")\n        g = simplify_dp(g, dp_eps)\n"
assert t.count(a) == 1
open(p, "w", encoding="utf-8").write(t.replace(a, b))
