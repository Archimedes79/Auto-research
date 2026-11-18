import os
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "evaluation/jmap_eval.py")
s = open(p, encoding="utf-8").read()
old = "median_ms=float(np.median(flat)), p90_ms"
assert s.count(old) == 1
s = s.replace(old, "n_timings=int(flat.size), median_ms=float(np.median(flat)), p90_ms")
open(p, "w", encoding="utf-8").write(s)
p = os.path.join(R, "training/grid/test_jmap_train.py")
s = open(p, encoding="utf-8").read()
old = 'assert tm_["K"] == 64 and tm_["frames"] == 3 and'
assert s.count(old) == 1
s = s.replace(old, 'assert tm_["K"] == 64 and tm_["frames"] == 3 and tm_["n_timings"] == 3 * 1 and')
old = "tz[\"params_block\"] == 0 and tz[\"gmacs_head\"]"
assert s.count(old) == 1
s = s.replace(old, "tz[\"n_timings\"] == 2 and tz[\"params_block\"] == 0 and tz[\"gmacs_head\"]")
open(p, "w", encoding="utf-8").write(s)
print("ok")
