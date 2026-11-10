p = "evalsuite/results.py"
s = open(p, encoding="utf-8").read()
old = "            arrays.append(np.ascontiguousarray(o))\n            return {\"$a\": len(arrays) - 1}"
new = "            arrays.append(o if o.flags[\"C_CONTIGUOUS\"] else np.array(o, order=\"C\"))      # (ascontiguousarray would turn a 0-d array into a 1-d one)\n            return {\"$a\": len(arrays) - 1}"
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
