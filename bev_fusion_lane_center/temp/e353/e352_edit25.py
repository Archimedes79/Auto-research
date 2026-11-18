p = "temp/e353/e352_mutants.py"
s = open(p, encoding="utf-8").read()
bad = '("group_means_misaligned", "val=calib.mean[tap][idx]))\n    return out", "val=calib.mean[tap][:len(idx)]))\n    return out", ["groups_variant_references"]),'
good = '("group_means_misaligned", "val=calib.mean[tap][idx]))\n    return out", "val=calib.mean[tap][:len(idx)]))\n    return out", ["groups_variant_references"]),'
assert bad in s
open(p, "w", encoding="utf-8").write(s.replace(bad, good))
