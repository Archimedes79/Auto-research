# E298: the driven flag of the true pairs in the batch (pdrv) and the weight of the positive pair terms (default off: behaviour unchanged)
p = "training/data/lane_gt_data.py"
s = open(p, encoding="utf-8").read()
old = "            jt, _ = QJ.junction_targets(g, paint)\n            out.update({n: torch.from_numpy(np.ascontiguousarray(v)) for n, v in jt.items()})\n"
new = ("            jt, jg = QJ.junction_targets(g, paint)\n            out.update({n: torch.from_numpy(np.ascontiguousarray(v)) for n, v in jt.items()})\n"
       "            npair = len(out[\"pairs\"]) if \"pairs\" in out else 0                                           # E298: 1 for a true pair the ego drives (tier 1), in the order of `pairs` (the sorted true pairs of `qends.end_table` = the rows of `jgt`'s pairs)\n"
       "            out[\"pdrv\"] = torch.from_numpy(np.ascontiguousarray(jg[\"pairs\"][:, 9].astype(np.float32))) if (jg is not None and len(jg[\"pairs\"]) == npair) else torch.zeros(npair)\n")
assert old in s
s = s.replace(old, new, 1)
old = "        out.update(ent=ent, ext=ext, ent_m=ent_m, ext_m=ext_m, pairs=pairs)\n"
new = ("        out.update(ent=ent, ext=ext, ent_m=ent_m, ext_m=ext_m, pairs=pairs)\n"
       "        if \"pdrv\" in b[0]:                                                                                # E298\n"
       "            pdrv = torch.zeros(B, P_)\n"
       "            for i, o in enumerate(b):\n"
       "                pdrv[i, :len(o[\"pdrv\"])] = o[\"pdrv\"]\n"
       "            out[\"pdrv\"] = pdrv\n")
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8").write(s)

q = "training/grid/jends.py"
t = open(q, encoding="utf-8").read()
old = "def jends_loss(out, batch, cross_w=1.0):"
new = "def jends_loss(out, batch, cross_w=1.0, driven_w=0.0):"
assert old in t
t = t.replace(old, new, 1)
old = "    allowed = out[\"allowed\"]\n    z = out[\"logit\"].float()\n    info = {}\n    if allowed.any():\n        bce = (F.binary_cross_entropy_with_logits(z, y, reduction=\"none\", pos_weight=torch.tensor(3.0, device=dev)) * allowed).sum() / allowed.sum()"
new = ("    allowed = out[\"allowed\"]\n    z = out[\"logit\"].float()\n    info = {}\n    wmat = torch.ones_like(y)\n"
       "    if driven_w > 0 and \"pdrv\" in batch:                                                                   # E298: the positive terms of the pairs the ego drives count 1 + driven_w times\n"
       "        Dv = torch.zeros(B, E + 1, X + 1, device=dev)\n"
       "        pd = batch[\"pdrv\"].to(dev)\n"
       "        Dv[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = pd[pv]\n"
       "        yd = Dv[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]\n"
       "        wmat = 1.0 + driven_w * yd * y\n"
       "    if allowed.any():\n"
       "        bce = (F.binary_cross_entropy_with_logits(z, y, reduction=\"none\", pos_weight=torch.tensor(3.0, device=dev)) * wmat * allowed).sum() / allowed.sum()")
assert old in t
t = t.replace(old, new, 1)
open(q, "w", encoding="utf-8").write(t)

r = "training/train/jends_train.py"
u = open(r, encoding="utf-8").read()
old = '    ap.add_argument("--cross-w", type=float, default=1.0)\n'
new = '    ap.add_argument("--cross-w", type=float, default=1.0)\n    ap.add_argument("--driven-w", type=float, default=0.0, help="E298: the positive pair terms of the pairs the ego drives count 1 + w times")\n'
assert old in u
u = u.replace(old, new, 1)
old = "        loss, info = JE.jends_loss(out, b, cross_w=a.cross_w)"
new = "        loss, info = JE.jends_loss(out, b, cross_w=a.cross_w, driven_w=a.driven_w)"
assert old in u
u = u.replace(old, new, 1)
open(r, "w", encoding="utf-8").write(u)
print("patched")
