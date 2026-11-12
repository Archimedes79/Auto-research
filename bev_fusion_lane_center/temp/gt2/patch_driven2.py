p = "training/grid/jends.py"
s = open(p, encoding="utf-8").read()
old = "        assert info[\"pos_pairs\"] >= 2 * B, info\n"
new = ("        assert info[\"pos_pairs\"] >= 2 * B, info\n"
       "        if order and evidence:                                                                         # E298: the driven weight raises the positive BCE terms and changes nothing when off\n"
       "            batch[\"pdrv\"] = torch.ones(B, 2)\n"
       "            _, i0 = jends_loss(out, batch, driven_w=0.0)\n"
       "            _, i2 = jends_loss(out, batch, driven_w=2.0)\n"
       "            assert i2[\"bce\"] > i0[\"bce\"] + 1e-6 and abs(i0[\"bce\"] - info[\"bce\"]) < 1e-6, (i0, i2, info)\n"
       "            batch[\"pdrv\"] = torch.zeros(B, 2)\n"
       "            _, iz = jends_loss(out, batch, driven_w=2.0)\n"
       "            assert abs(iz[\"bce\"] - i0[\"bce\"]) < 1e-6, (iz, i0)\n")
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
