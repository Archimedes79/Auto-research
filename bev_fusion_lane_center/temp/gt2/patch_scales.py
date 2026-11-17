# E303: two learned Hermite handle scales per pair (additive, default off)
def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:60])
    s = s.replace(old, new, count)
    open(path, "w", encoding="utf-8").write(s)


# 1 qends.curve_s
sub("training/grid/qends.py", "def _selftest():", '''def curve_s(a, ta, b, tb, sa=1.0 / 3.0, sb=1.0 / 3.0, step=0.25):
    """the cubic Bezier / Hermite curve from a (heading ta) to b (heading tb) with the handle scales sa, sb in units of the chord (the control points a + sa x chord x ta and b - sb x chord x tb; sa = sb = 1/3 is `curve` with k = 1;
    the scales of `qtargets.fit_long_scales` / E303 are these), points about `step` metres apart"""
    L = float(np.linalg.norm(b - a))
    n = max(8, int(1.6 * L / step))
    s = np.linspace(0, 1, n)[:, None]
    q1, q2 = a + sa * L * ta, b - sb * L * tb
    return (1 - s) ** 3 * a + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * b


def _selftest():''')
sub("training/grid/qends.py", "    c = curve(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))\n", "    c = curve(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))\n    c2 = curve_s(np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([4.0, 4.0]), np.array([1.0, 0.0]))\n    assert np.allclose(c, c2), 'curve_s with the scales 1/3 is curve with k = 1'\n")

# 2 qends_eval: pairs may carry the scales
sub("evaluation/qends_eval.py", "            if any(np.linalg.norm(P[0] - pa) <= COVER_POS and np.linalg.norm(P[-1] - pb) <= COVER_POS for pa, ta, pb, tb in pairs):", "            if any(np.linalg.norm(P[0] - pr[0]) <= COVER_POS and np.linalg.norm(P[-1] - pr[2]) <= COVER_POS for pr in pairs):")
sub("evaluation/qends_eval.py", "    for n, (pa, ta, pb, tb) in enumerate(pairs):\n        m = len(L2)\n        L2.append(QEND.curve(np.asarray(pa, float), np.asarray(ta, float), np.asarray(pb, float), np.asarray(tb, float), 1.0, 0.25))",
    "    for n, pr in enumerate(pairs):\n        pa, ta, pb, tb = pr[:4]                                                                       # E303: a pair may carry the two handle scales (pr[4], pr[5])\n        m = len(L2)\n        if len(pr) > 4:\n            L2.append(QEND.curve_s(np.asarray(pa, float), np.asarray(ta, float), np.asarray(pb, float), np.asarray(tb, float), float(pr[4]), float(pr[5])))\n        else:\n            L2.append(QEND.curve(np.asarray(pa, float), np.asarray(ta, float), np.asarray(pb, float), np.asarray(tb, float), 1.0, 0.25))")
sub("evaluation/qends_eval.py", "        pairs = [(e[\"kp_pos\"][i], e[\"kp_dir\"][i], e[\"kp_pos\"][k + j], e[\"kp_dir\"][k + j]) for i, j in zip(ia, ib)]",
    "        pairs = [(e[\"kp_pos\"][i], e[\"kp_dir\"][i], e[\"kp_pos\"][k + j], e[\"kp_dir\"][k + j]) + (tuple(e[\"scale\"][i, j]) if e.get(\"scale\") is not None and not a.chord else ()) for i, j in zip(ia, ib)]")
sub("evaluation/qends_eval.py", "    s.add_argument(\"--base\", default=None)", "    s.add_argument(\"--base\", default=None)\n    s.add_argument(\"--chord\", action=\"store_true\", help=\"E303: use the chord tangent for the connectors even when the dump carries learned handle scales (the control)\")")

# 3 jgt.pair_pts
sub("training/grid/jgt.py", "    pairs = np.array(rows, float)\n", "    pairs = np.array(rows, float)\n    first = {}\n    for r, a_, b_ in zip(runs, eid, xid):\n        first.setdefault((a_, b_), r[\"pts\"])\n    pair_pts = [first[(int(r[0]), int(r[1]))] for r in rows]                                                         # E303: the GT connector points of every pair (first run), in the order of `pairs`\n")
sub("training/grid/jgt.py", "                ent_vis=ent_vis, ext_vis=ext_vis, core=core,", "                ent_vis=ent_vis, ext_vis=ext_vis, pair_pts=pair_pts, core=core,")

# 4 qjunction.fit_pair_scales
sub("training/grid/qjunction.py", "def painted_flags(", '''def fit_pair_scales(jg, scales=(0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):
    """E303: the two handle scales (sa, sb, in chords) of the Hermite curve of every true pair, fitted to the GT connector points of the pair (grid search, the cost of `qtargets.fit_long_scales`) -> (P, 2) float32 in the order of
    `jg['pairs']` (NaN when a pair has no connector points). What the pair head's scale outputs are trained towards."""
    import qends as QE
    from scipy.spatial import cKDTree
    out = np.full((0 if jg is None else len(jg["pairs"]), 2), np.nan, np.float32)
    if jg is None:
        return out
    for n, (r, pts) in enumerate(zip(jg["pairs"], jg["pair_pts"])):
        a, b = int(r[0]), int(r[1])
        pa, ta, pb, tb = jg["ent"][a, :2], jg["ent"][a, 2:], jg["ext"][b, :2], jg["ext"][b, 2:]
        tgt = np.asarray(pts, float)
        if len(tgt) < 3 or np.linalg.norm(pb - pa) < 1.0:
            continue
        tt = cKDTree(tgt)
        best = 1e9
        for sa in scales:
            for sb in scales:
                cur = QE.curve_s(pa, ta, pb, tb, sa, sb, step=0.5)
                m = tt.query(cur)[0].mean() + 0.2 * cKDTree(cur).query(tgt)[0].mean()
                if m < best:
                    best, out[n] = m, (sa, sb)
    return out


def painted_flags(''')
sub("training/grid/qjunction.py", "    pe, px = painted_flags(jg, paint)\n    assert len(pe)", "    sc = fit_pair_scales(jg)\n    assert sc.shape == (len(jg[\"pairs\"]), 2) and np.isfinite(sc).all() and (sc > 0.05).all(), sc\n    pe, px = painted_flags(jg, paint)\n    assert len(pe)")

# 5 lane_gt_data pscale
sub("training/data/lane_gt_data.py", "            out[\"pdrv\"] = torch.from_numpy(", "            out[\"pscale\"] = torch.from_numpy(QJ.fit_pair_scales(jg)) if (jg is not None and len(jg[\"pairs\"]) == npair) else torch.full((npair, 2), float(\"nan\"))        # E303: the fitted handle scales of the true pairs\n            out[\"pdrv\"] = torch.from_numpy(")
sub("training/data/lane_gt_data.py", "            out[\"pdrv\"] = pdrv\n", "            out[\"pdrv\"] = pdrv\n            ps = torch.full((B, P_, 2), float(\"nan\"))                                                         # E303\n            for i, o in enumerate(b):\n                ps[i, :len(o[\"pscale\"])] = o[\"pscale\"]\n            out[\"pscale\"] = ps\n")

# 6 jends: scales option and loss
sub("training/grid/jends.py", "    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64):\n        super().__init__()\n        self.k, self.order, self.evidence = k, order, evidence\n",
    "    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False):\n        super().__init__()\n        self.k, self.order, self.evidence, self.scales = k, order, evidence, scales\n")
sub("training/grid/jends.py", "        self.pair = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))\n", "        self.pair = nn.Sequential(nn.Linear(nin, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))\n        if scales:                                                                                           # E303: the two handle scales of the delivered curve, in chords (0.1 .. 0.85 like the fit grid)\n            self.scale = nn.Sequential(nn.Linear(nin, 32), nn.GELU(), nn.Linear(32, 2))\n")
sub("training/grid/jends.py", "        logit = self.pair(torch.cat(feats, -1))[..., 0]\n        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)\n",
    "        fcat = torch.cat(feats, -1)\n        logit = self.pair(fcat)[..., 0]\n        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)\n        if self.scales:\n            out[\"scale\"] = 0.1 + 0.75 * self.scale(fcat).sigmoid()\n")
sub("training/grid/jends.py", "def jends_loss(out, batch, cross_w=1.0, driven_w=0.0):", "def jends_loss(out, batch, cross_w=1.0, driven_w=0.0, scale_w=0.5):")
sub("training/grid/jends.py", "    if \"ga\" in out and cross_w > 0 and allowed.any():", '''    if "scale" in out and "pscale" in batch and scale_w > 0:                                                    # E303: L1 of the predicted handle scales against the fitted ones, on the matched true pairs
        S = torch.full((B, E + 1, X + 1, 2), float("nan"), device=dev)
        ps = batch["pscale"].to(dev)
        S[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = ps[pv]
        St = S[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]
        m = (y > 0) & allowed & torch.isfinite(St[..., 0])
        if m.any():
            ls = (out["scale"][m] - St[m]).abs().mean()
            loss = loss + scale_w * ls
            info["scale_l1"] = float(ls)
    if "ga" in out and cross_w > 0 and allowed.any():''')

# 7 trainer / eval
sub("training/train/jends_train.py", "    def __init__(self, maps_args, order=True, evidence=True, d=64):", "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False):")
sub("training/train/jends_train.py", "self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=maps_args.get(\"d\", 64))", "self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=maps_args.get(\"d\", 64), scales=scales)")
sub("training/train/jends_train.py", "    ap.add_argument(\"--driven-w\"", "    ap.add_argument(\"--scales\", action=\"store_true\", help=\"E303: two handle scales per pair (the delivered curve), trained against the fitted scales of the true pairs\")\n    ap.add_argument(\"--driven-w\"")
sub("training/train/jends_train.py", "model = JEndModel(ck[\"args\"], order=not a.no_order, evidence=not a.no_evidence)", "model = JEndModel(ck[\"args\"], order=not a.no_order, evidence=not a.no_evidence, scales=a.scales)")
sub("evaluation/jends_eval.py", "m = JT.JEndModel(a[\"maps_args\"], order=not a[\"no_order\"], evidence=not a[\"no_evidence\"])", "m = JT.JEndModel(a[\"maps_args\"], order=not a[\"no_order\"], evidence=not a[\"no_evidence\"], scales=bool(a.get(\"scales\", False)))")
sub("evaluation/jends_eval.py", "valid=out[\"valid\"][i].cpu().numpy(), logit=out[\"logit\"][i].float().cpu().numpy(), allowed=out[\"allowed\"][i].cpu().numpy()))", "valid=out[\"valid\"][i].cpu().numpy(), logit=out[\"logit\"][i].float().cpu().numpy(), allowed=out[\"allowed\"][i].cpu().numpy(),\n                                   scale=out[\"scale\"][i].float().cpu().numpy() if \"scale\" in out else None))")
print("patched")
