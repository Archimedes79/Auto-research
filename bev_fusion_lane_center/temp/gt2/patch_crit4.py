def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


p = "training/grid/jends.py"
# ---- 1: the plan is a mass; the pair probability is the plan entry over the smaller capacity (a proper [0, 1] number, no zero-gradient clamp for false pairs with P > 1)
sub(p, "E302 `capacity`: instead of independent pair logits (+ dual softmax) the pair probability is the plan of a log-domain Sinkhorn with a dustbin:", "E302 `capacity`: instead of independent pair logits (+ dual softmax) the pair probability comes from the plan of a log-domain Sinkhorn with a dustbin:")
sub(p, "The plan is read as a logit (`logit = log P / (1 - P)`), so the read-out (`logit >= 0`), the counts and the crossing penalty work unchanged.",
    "A plan entry is a MASS (up to the degree), the pair probability is q = P / max(1, min(row mass, column mass)) in [0, 1] (the share of the smaller capacity this pair uses; one entry of degree 1 with two equally good exits gives 0.5 each), read as a logit, so the read-out (`logit >= 0`, i.e. q >= 0.5), the counts and the crossing penalty work unchanged.")
sub(p, "def sinkhorn_plan(z, allowed, re, rx, alpha, iters=SINK_ITERS):", "def sinkhorn_plan(z, allowed, re, rx, alpha, iters=SINK_ITERS, full=False):")
sub(p, "differentiable in z, re, rx and alpha. P_ij is about the probability of the pair when the degrees are right.\"\"\"", "differentiable in z, re, rx and alpha. P_ij is a mass (<= min(re_i, rx_j)); `full`: the whole (B, K + 1, K + 1) plan with the dustbin row / column.\"\"\"")
sub(p, "    return (M + u[:, :, None] + v[:, None, :])[:, :K_, :K_].exp()\n", "    plan = (M + u[:, :, None] + v[:, None, :]).exp()\n    return plan if full else plan[:, :K_, :K_]\n")
sub(p, "            de, dx = self.deg_e(ta_t), self.deg_x(tb_t)\n", "            de, dgx = self.deg_e(ta_t), self.deg_x(tb_t)\n")
sub(p, "            rx = (dx.float().softmax(-1) * vals).sum(-1) * valid[:, 1]\n            Pm = sinkhorn_plan(logit.float(), allowed, re, rx, self.alpha).clamp(1e-5, 1 - 1e-5)\n            logit = (Pm / (1 - Pm)).log()\n            extra = dict(deg_e=de, deg_x=dx)",
    "            rx = (dgx.float().softmax(-1) * vals).sum(-1) * valid[:, 1]\n            Pm = sinkhorn_plan(logit.float(), allowed, re, rx, self.alpha)\n            Pm = (Pm / torch.minimum(re[:, :, None], rx[:, None, :]).clamp_min(1.0)).clamp(1e-5, 1 - 1e-5)          # the share of the smaller capacity\n            logit = (Pm / (1 - Pm)).log()\n            extra = dict(deg_e=de, deg_x=dgx)")
# ---- 2: the degree classes as a function (testable), `dual` logged only when it is in the loss
sub(p, "def jends_loss(out, batch, cross_w=1.0, driven_w=0.0, scale_w=0.5):", '''def degree_classes(T, match_e, match_x):
    """(cls_e, cls_x) (B, K) long: the degree class {0, 1, 2, 3+} of every keypoint: the number of true pairs of the GT end it is matched to (rows of T (B, E, X) = out-degree of an entry, columns = in-degree of an exit), 0 for an unmatched keypoint"""
    res = []
    for deg, mt in ((T.sum(2), match_e), (T.sum(1), match_x)):
        res.append(torch.where(mt >= 0, deg.gather(1, mt.clamp_min(0)).clamp(max=3).long(), torch.zeros_like(mt)))
    return res


def jends_loss(out, batch, cross_w=1.0, driven_w=0.0, scale_w=0.5):''')
sub(p, '''        Tt = T[:, :E, :X].float()
        ce_tot, acc_tot, n_tot = z.new_zeros(()), 0.0, 0
        for k_, (deg, mt, dl) in enumerate(((Tt.sum(2), match[0], out["deg_e"]), (Tt.sum(1), match[1], out["deg_x"]))):
            cls = torch.where(mt >= 0, deg.gather(1, mt.clamp_min(0)).clamp(max=3).long(), torch.zeros_like(mt))
            v = out["valid"][:, k_]''', '''        ce_tot, acc_tot, n_tot = z.new_zeros(()), 0.0, 0
        cls_all = degree_classes(T[:, :E, :X].float(), match[0], match[1])
        for k_, (cls, dl) in enumerate(zip(cls_all, (out["deg_e"], out["deg_x"]))):
            v = out["valid"][:, k_]''')
sub(p, "    info.update(bce=float(bce), dual=float(dual), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))\n    if cap:", "    info.update(bce=float(bce), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))\n    if not cap:\n        info[\"dual\"] = float(dual)\n    if cap:")
# ---- 3: the self-test pins the capacity path (E != X, sum(re) != sum(rx), the class targets)
sub(p, "    mc = JEnds(d=d, order=True, evidence=True, capacity=True)\n", '''    # critic 2026-11-12: pairs 0 -> 0, 0 -> 1, 1 -> 2, 2 -> 2 (entry 0 has degree 2, exit 2 a merge of two): the class targets, with an unmatched keypoint
    Tm = torch.zeros(1, 3, 3)
    for a_, b_ in ((0, 0), (0, 1), (1, 2), (2, 2)):
        Tm[0, a_, b_] = 1.0
    ce_, cx_ = degree_classes(Tm, torch.tensor([[0, 1, 2, -1]]), torch.tensor([[0, 1, 2, -1]]))
    assert ce_.tolist() == [[2, 1, 1, 0]] and cx_.tolist() == [[1, 1, 2, 0]], (ce_, cx_)
    # the full plan has the marginals it was given, also when the total entry mass is not the total exit mass (E != X), and the share of the smaller capacity is in [0, 1]
    ze = torch.randn(1, 5, 5)
    re_, rx_ = torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.5, 0.0, 0.0]])
    full = sinkhorn_plan(ze, torch.ones(1, 5, 5, dtype=torch.bool), re_, rx_, torch.zeros(()), iters=50, full=True)
    assert abs(float(full[0, 0].sum()) - 2.0) < 1e-2 and abs(float(full[0, 1].sum()) - 1.0) < 1e-2 and abs(float(full[0, 5].sum()) - 2.5) < 1e-2, full[0].sum(1)
    assert abs(float(full[0, :, 0].sum()) - 1.0) < 1e-2 and abs(float(full[0, :, 5].sum()) - 3.0) < 1e-2, full[0].sum(0)
    mc = JEnds(d=d, order=True, evidence=True, capacity=True)
''')

# ---- 4: the _maps checkpoint keeps the raster, jmaps_eval sets the raster explicitly (legacy unsets an inherited exact)
p = "training/train/jends_train.py"
sub(p, "args=dict(ck[\"args\"], e2e_from=a.name))", "args=dict(ck[\"args\"], e2e_from=a.name, raster=a.raster))")
p = "evaluation/jmaps_eval.py"
sub(p, "    if raster == \"exact\":\n        os.environ[\"BFLC_RASTER\"] = \"exact\"\n", "    if raster == \"exact\":\n        os.environ[\"BFLC_RASTER\"] = \"exact\"\n    else:\n        os.environ.pop(\"BFLC_RASTER\", None)                                                               # an inherited exact setting must not score a legacy run\n")

# ---- 5: thin paint only gets the low threshold; the docs say what the mode is (a supersampled area rule, about 3 % of the cells differ from the analytic one)
p = "training/grid/jgt.py"
sub(p, "(`line`: more than a quarter, the thin paint)\"\"\"", "(`line`: more than a quarter for polygons thinner than one cell, so a thin stop line stays visible; the others use the half rule). 8 x 8 supersampling: about 3 % of the cells differ from the analytic area rule (the name 'exact' is relative to the legacy fill)\"\"\"")
sub(p, "        cov = big.reshape(r1 - r0, SS, c1 - c0, SS).mean((1, 3))\n        mask[r0:r1, c0:c1] |= (cov > (0.25 if line else 0.5)).astype(mask.dtype)",
    "        cov = big.reshape(r1 - r0, SS, c1 - c0, SS).mean((1, 3))\n        per = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum()\n        thick = 2.0 * abs(0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])) / max(per, 1e-9)         # 2 A / perimeter ~ the width of a strip\n        mask[r0:r1, c0:c1] |= (cov > (0.25 if (line and thick < 1.0) else 0.5)).astype(mask.dtype)")

# ---- 6: jends_eval: the pyramid check is not vacuous; jpair_pr: the exact precision at a recall from the sorted scores
p = "evaluation/jends_eval.py"
sub(p, "    pyr_same = all(", "    keys = [k for k in base if k.startswith(\"pyr.\") and ((\"jmodel.\" + k) in ck[\"state\"] or k in ck[\"state\"])]\n    assert keys, \"no pyramid tensor of the base run is in the checkpoint: the pyramid check would be empty\"\n    pyr_same = all(")
p = "evaluation/jpair_pr.py"
sub(p, "def candidates(run, kp_thr):", '''def precision_at_recall_exact(scores, labels, n_gt, recall):
    """the precision at the smallest number of delivered pairs (the best scores first) whose recall reaches `recall`: no threshold grid (None when the candidates never reach it)"""
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[order])
    k = np.searchsorted(tp, np.ceil(recall * n_gt - 1e-9))
    return None if k >= len(tp) or n_gt == 0 or recall <= 0 else float(tp[k] / (k + 1))


def candidates(run, kp_thr):''')
sub(p, "        pr = precision_at_recall(curve(*cands[a.run][:2], cands[a.run][2]), rc)\n", "        pr = precision_at_recall_exact(cands[a.run][0], cands[a.run][1], cands[a.run][2], rc)\n")
sub(p, "    assert precision_at_recall(rows, 0.4) == 1.0 and precision_at_recall(rows, 0.6) == 0.75 and precision_at_recall(rows, 0.9) is None\n", "    assert precision_at_recall(rows, 0.4) == 1.0 and precision_at_recall(rows, 0.6) == 0.75 and precision_at_recall(rows, 0.9) is None\n    assert precision_at_recall_exact(sc, lab, 5, 0.4) == 1.0 and abs(precision_at_recall_exact(sc, lab, 5, 0.6) - 0.75) < 1e-12 and precision_at_recall_exact(sc, lab, 5, 0.9) is None   # 0.6 = three true pairs: reached at the fourth delivered\n")
print("ok")
