p = "training/data/lane_gt_data.py"
s = open(p, encoding="utf-8").read()
s = s.replace("        lk, lk_long = QT.link_array(nt)\n", "        lk, lk_long = QT.link_array(nt)\n        lk_scale = QT.fit_long_scales(nt).astype(np.float32)\n")
s = s.replace("out = dict(x=torch.from_numpy(x), row=row, k=k, token=token, lk=torch.from_numpy(lk), lk_long=torch.from_numpy(lk_long),", "out = dict(x=torch.from_numpy(x), row=row, k=k, token=token, lk=torch.from_numpy(lk), lk_long=torch.from_numpy(lk_long), lk_scale=torch.from_numpy(lk_scale),")
s = s.replace("""    lk, lkl = -torch.ones(B, L, 4, dtype=torch.long), torch.zeros(B, L, dtype=torch.bool)
    for i, o in enumerate(b):
        n = len(o["lk"])
        lk[i, :n], lkl[i, :n] = o["lk"], o["lk_long"]
    out.update(lk=lk, lk_long=lkl)""", """    lk, lkl, lks = -torch.ones(B, L, 4, dtype=torch.long), torch.zeros(B, L, dtype=torch.bool), torch.full((B, L, 2), float("nan"))
    for i, o in enumerate(b):
        n = len(o["lk"])
        lk[i, :n], lkl[i, :n], lks[i, :n] = o["lk"], o["lk_long"], o["lk_scale"]
    out.update(lk=lk, lk_long=lkl, lk_scale=lks)""")
s = s.replace("links `lk` (L, 4) int cells (iu, ju, iv, jv) + `lk_long`;", "links `lk` (L, 4) int cells (iu, ju, iv, jv) + `lk_long` + `lk_scale` (L, 2) the fitted Hermite handle scales of the long links (NaN for short ones);")
open(p, "w", encoding="utf-8").write(s)
