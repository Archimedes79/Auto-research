p = "temp/loader/loader_identity.py"
s = open(p, encoding="utf-8").read()
a = "    if torch.is_tensor(a):\n        return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)\n"
assert s.count(a) == 1
b = ("    if torch.is_tensor(a):\n        if not (torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype):\n            return False\n"
     "        if a.is_floating_point():                                                                        # NaN padding (pscale, ppts, lk_scale) is equal to NaN\n"
     "            return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a, nan=0.0), torch.nan_to_num(b, nan=0.0))\n"
     "        return torch.equal(a, b)\n")
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
