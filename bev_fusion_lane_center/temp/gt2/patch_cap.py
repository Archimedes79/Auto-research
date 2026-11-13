def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))


p = "training/grid/jends.py"
# ---- header text
sub(p, "Run the self-test: python jends.py\n\"\"\"", '''E302 `capacity`: instead of independent pair logits (+ dual softmax) the pair probability is the plan of a log-domain Sinkhorn with a dustbin: every entry has a predicted out-degree {0, 1, 2, 3+} (0 = a false keypoint), every exit an in-degree; an entry that
feeds a turn AND the straight exit (27 % of the entries, E285) has mass 2, a merge exit takes two entries (33 %). The plan is read as a logit (`logit = log P / (1 - P)`), so the read-out (`logit >= 0`), the counts and the crossing penalty work unchanged. Differentiable (10 unrolled iterations).
Run the self-test: python jends.py
"""''')
sub(p, "COS_TURN, MIN_CHORD = QEN.COS_TURN, QEN.MIN_CHORD\n", "COS_TURN, MIN_CHORD = QEN.COS_TURN, QEN.MIN_CHORD\nSINK_ITERS = 10\nDEG_VALUES = (0.0, 1.0, 2.0, 3.3)                                                                          # the degree classes 0, 1, 2, 3+ as masses\n")
# ---- the plan function before class JEnds
sub(p, "class JEnds(nn.Module):", '''def sinkhorn_plan(z, allowed, re, rx, alpha, iters=SINK_ITERS):
    """(B, K, K) transport plan of the pair scores `z` (B, K, K; disallowed pairs get ~no mass) with a dustbin row and column: row masses `re` (B, K) = the expected out-degree of the entries, column masses `rx` = the in-degree of the exits;
    the dustbin takes what is not connected (its mass makes both marginal sums equal). Log domain, `iters` unrolled iterations, differentiable in z, re, rx and alpha. P_ij is about the probability of the pair when the degrees are right."""
    B, K_ = z.shape[0], z.shape[1]
    S = torch.where(allowed, z, torch.full_like(z, -30.0))
    col = alpha.expand(B, K_, 1)
    M = torch.cat([torch.cat([S, col], 2), torch.cat([alpha.expand(B, 1, K_), z.new_zeros(B, 1, 1)], 2)], 1)             # (B, K + 1, K + 1)
    log_a = torch.cat([(re + 1e-6).log(), (rx.sum(1, keepdim=True) + 1e-6).log()], 1)
    log_b = torch.cat([(rx + 1e-6).log(), (re.sum(1, keepdim=True) + 1e-6).log()], 1)
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iters):
        u = log_a - torch.logsumexp(M + v[:, None, :], 2)
        v = log_b - torch.logsumexp(M + u[:, :, None], 1)
    return (M + u[:, :, None] + v[:, None, :])[:, :K_, :K_].exp()


class JEnds(nn.Module):''')
sub(p, "    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False):\n        super().__init__()\n        self.k, self.order, self.evidence, self.scales = k, order, evidence, scales\n",
    "    def __init__(self, d=64, hid=64, k=K, order=True, evidence=True, f1=64, scales=False, capacity=False):\n        super().__init__()\n        self.k, self.order, self.evidence, self.scales, self.capacity = k, order, evidence, scales, capacity\n")
sub(p, "        if scales:                                                                                           # E303:", "        if capacity:                                                                                         # E302: the degree heads on the keypoint tokens (classes 0, 1, 2, 3+) and the dustbin score\n            self.deg_e, self.deg_x = nn.Linear(d, 4), nn.Linear(d, 4)\n            self.alpha = nn.Parameter(torch.zeros(()))\n        if scales:                                                                                           # E303:")
sub(p, "        logit = self.pair(fcat)[..., 0]\n        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)\n",
    '''        logit = self.pair(fcat)[..., 0]
        extra = {}
        if self.capacity:                                                                                    # E302: the pair probability is the plan of a capacitated Sinkhorn, expressed as a logit
            de, dx = self.deg_e(ta_t), self.deg_x(tb_t)
            vals = logit.new_tensor(DEG_VALUES)
            re = (de.float().softmax(-1) * vals).sum(-1) * valid[:, 0]
            rx = (dx.float().softmax(-1) * vals).sum(-1) * valid[:, 1]
            Pm = sinkhorn_plan(logit.float(), allowed, re, rx, self.alpha).clamp(1e-5, 1 - 1e-5)
            logit = (Pm / (1 - Pm)).log()
            extra = dict(deg_e=de, deg_x=dx)
        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed, **extra)
''')
# ---- loss: degree cross-entropy, no dual softmax under capacity
sub(p, "    loss = bce + 0.5 * dual\n    info.update(bce=float(bce), dual=float(dual), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))\n",
    '''    cap = "deg_e" in out
    loss = bce + (0.0 if cap else 0.5 * dual)                                                               # the plan already makes rows and columns compete (E302)
    info.update(bce=float(bce), dual=float(dual), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))
    if cap:                                                                                                  # E302: the degree classes of the keypoints (a keypoint that is no GT end: class 0)
        Tt = T[:, :E, :X].float()
        ce_tot, acc_tot, n_tot = z.new_zeros(()), 0.0, 0
        for k_, (deg, mt, dl) in enumerate(((Tt.sum(2), match[0], out["deg_e"]), (Tt.sum(1), match[1], out["deg_x"]))):
            cls = torch.where(mt >= 0, deg.gather(1, mt.clamp_min(0)).clamp(max=3).long(), torch.zeros_like(mt))
            v = out["valid"][:, k_]
            if v.any():
                ce_tot = ce_tot + F.cross_entropy(dl.float()[v], cls[v])
                acc_tot += float((dl[v].argmax(-1) == cls[v]).float().mean())
                n_tot += 1
        if n_tot:
            loss = loss + 0.5 * ce_tot
            info.update(deg_ce=float(ce_tot), deg_acc=acc_tot / n_tot)
''')

# ---- the self-test
sub(p, "    # critic 2026-11-11: with no real peak at all", '''    # E302: the capacitated plan. Two entries, two exits, all degree 1, scores favour the diagonal: the plan is the identity; an entry of degree 2 feeds two exits of degree 1; a false keypoint (degree 0) gets no mass
    z = torch.full((1, 4, 4), -3.0)
    z[0, 0, 0], z[0, 1, 1] = 4.0, 4.0
    al = torch.zeros(())
    Pl = sinkhorn_plan(z, torch.ones(1, 4, 4, dtype=torch.bool), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), al)
    assert Pl[0, 0, 0] > 0.9 and Pl[0, 1, 1] > 0.9 and Pl[0, 0, 1] < 0.1 and Pl[0, 2:].sum() < 1e-3 and Pl[0, :, 2:].sum() < 1e-3, Pl[0]
    z2 = torch.full((1, 4, 4), -3.0)
    z2[0, 0, 0], z2[0, 0, 1] = 2.0, 2.0
    P2 = sinkhorn_plan(z2, torch.ones(1, 4, 4, dtype=torch.bool), torch.tensor([[2.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 0.0, 0.0]]), al)
    assert P2[0, 0, 0] > 0.8 and P2[0, 0, 1] > 0.8, P2[0]
    mc = JEnds(d=d, order=True, evidence=True, capacity=True)
    out = mc(jout, P, gt=gt)
    assert out["deg_e"].shape == (B, K, 4) and torch.isfinite(out["logit"]).all()
    loss, info = jends_loss(out, batch)
    loss.backward()
    dead = [n for n, p_ in mc.named_parameters() if p_.grad is None]
    assert not dead and "deg_ce" in info and info["pos_pairs"] >= 2 * B, (dead, info)
    # critic 2026-11-11: with no real peak at all''')

# ---- jends_train: --capacity
p = "training/train/jends_train.py"
sub(p, "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False, e2e=False):", "    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False, e2e=False, capacity=False):")
sub(p, "self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=maps_args.get(\"d\", 64), scales=scales)", "self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=maps_args.get(\"d\", 64), scales=scales, capacity=capacity)")
sub(p, '    ap.add_argument("--e2e", action="store_true"', '    ap.add_argument("--capacity", action="store_true", help="E302: degree heads + a capacitated Sinkhorn plan with a dustbin instead of independent pair logits + dual softmax")\n    ap.add_argument("--e2e", action="store_true"')
sub(p, "scales=a.scales, e2e=a.e2e)", "scales=a.scales, e2e=a.e2e, capacity=a.capacity)")
sub(p, "evidence={not a.no_evidence} e2e={a.e2e}:", "evidence={not a.no_evidence} e2e={a.e2e} capacity={a.capacity}:")
sub(p, "e2e=a.e2e, maps=mp))", "e2e=a.e2e, capacity=a.capacity, maps=mp))")
p = "evaluation/jends_eval.py"
sub(p, 'scales=bool(a.get("scales", False)))', 'scales=bool(a.get("scales", False)), capacity=bool(a.get("capacity", False)))')
print("ok")
