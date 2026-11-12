p = "training/grid/qgraph.py"
s = open(p, encoding="utf-8").read()
old = "    y = L[\"label\"]\n    npos = y.sum().clamp_min(1.0)\n    info = {}\n    l_b = focal_bce(L[\"logit\"].float(), y, norm=npos)"
new = """    y = L["label"]
    npos = y.sum().clamp_min(1.0)
    info = {}
    lg = L["is_long"]
    sh = ~lg
    l_b = focal_bce(L["logit"].float()[sh], y[sh], norm=y[sh].sum().clamp_min(1.0))               # short links: focal BCE over all candidate pairs
    l_long = torch.zeros((), device=dev)
    if lg.any():                                                                                  # long links: their positives are ~5 % of all positives and ~0.3 % of their own candidates: a balanced BCE of their own
        yl = y[lg]
        npl, nnl = yl.sum(), (1 - yl).sum()
        wgt = torch.where(yl > 0.5, (nnl / npl.clamp_min(1.0)).clamp(1.0, 80.0), torch.ones_like(yl))
        l_long = (F.binary_cross_entropy_with_logits(L["logit"].float()[lg], yl, reduction="none") * wgt).sum() / (wgt.sum() + 1.0)"""
assert old in s
s = s.replace(old, new)
s = s.replace("    info.update(link=float(l_b), deg=float(l_deg), scale=float(l_sc), pos=int(npos), pairs=len(y), miss=int(L.get(\"gt_missing\", 0)))\n    return l_b + 2.0 * l_sc, info",
              "    info.update(link=float(l_b), long=float(l_long), deg=float(l_deg), scale=float(l_sc), pos=int(npos), pairs=len(y), miss=int(L.get(\"gt_missing\", 0)))\n    return l_b + 2.0 * l_long + 2.0 * l_sc, info")
open(p, "w", encoding="utf-8").write(s)
