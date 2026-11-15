import re
p = "temp/e320/dev/jends.pysrc"
s = open(p, encoding="utf-8").read()
def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old, s.count(old))
    s = s.replace(old, new)
rep('''    kp, kd = out["kp_pos"].float(), out["kp_dir"].float()
    info, loss = {}, None
    if m.any():
        b_, i_, j_ = m.nonzero(as_tuple=True)
        sc = out.get("scale")
        curve = _bezier_s(kp[b_, i_], kd[b_, i_], kp[b_, K_ + j_], kd[b_, K_ + j_],''', '''    kp, kd = out["kp_pos"].float(), out["kp_dir"].float()
    ep, ed = (out["att_pos"].float(), out["att_dir"].float()) if "att_pos" in out else (kp, kd)               # E320: the FINAL ends of the delivered curve are the attached ones
    info, loss = {}, None
    if m.any():
        b_, i_, j_ = m.nonzero(as_tuple=True)
        sc = out.get("scale")
        curve = _bezier_s(ep[b_, i_], ed[b_, i_], ep[b_, K_ + j_], ed[b_, K_ + j_],''')
rep('''    ends, n_end, d_all = kp.new_zeros(()), 0, []
''', '''    ends, n_end, d_all = kp.new_zeros(()), 0, []
    ce_tot, n_acc, n_bin, kp_d = kp.new_zeros(()), 0, 0, []
''')
rep('''        d = ((kpk - g) ** 2).sum(-1).add(1e-8).sqrt()[ok]
''', '''        epk = ep[:, k * K_:(k + 1) * K_]
        d = ((epk - g) ** 2).sum(-1).add(1e-8).sqrt()[ok]
        if "att_logit" in out:                                                                               # E320: the attachment target of a keypoint = the node of B's head nearest to the GT end it belongs to (within ATT_TGT m), else 'no attachment'
            lg = out["att_logit"][:, k * K_:(k + 1) * K_]
            with torch.no_grad():
                dc = ((out["att_cpos"][:, k * K_:(k + 1) * K_] - g[:, :, None, :]) ** 2).sum(-1).sqrt()
                dc = torch.where(out["att_cvalid"][:, k * K_:(k + 1) * K_], dc, torch.full_like(dc, 1e6))
                tgt = torch.where(dc.min(2)[0] <= ATT_TGT, dc.argmin(2) + 1, torch.zeros_like(dc[..., 0], dtype=torch.long))
            ce_tot = ce_tot + F.cross_entropy(lg[ok], tgt[ok], reduction="sum")
            n_acc += int((lg[ok].argmax(-1) == tgt[ok]).sum())
            n_bin += int((tgt[ok] == 0).sum())
            kp_d.append(((kpk - g) ** 2).sum(-1).add(1e-8).sqrt()[ok].detach())
''')
rep('''        ends = ends / n_end
        loss = ends if loss is None else loss + ends
        info.update(geom_end=float(ends), geom_end_d=float(torch.cat(d_all).mean()))''', '''        ends = ends / n_end
        loss = ends if loss is None else loss + ends
        info.update(geom_end=float(ends), geom_end_d=float(torch.cat(d_all).mean()))
        if kp_d:                                                                                             # E320: the cross entropy of the attachment over the candidates (+ the dustbin)
            ce = ce_tot / n_end
            loss = loss + ce
            info.update(att_ce=float(ce), att_acc=n_acc / n_end, att_bin=n_bin / n_end, att_kp_d=float(torch.cat(kp_d).mean()))''')
rep('''    (E / X = none) as in `jends_loss` (they pick the GT polyline of a keypoint pair)."""''', '''    (E / X = none) as in `jends_loss` (they pick the GT polyline of a keypoint pair).
    E320 (`att_pos` in out): the curve and the end term use the ATTACHED ends (`Attach`), and the attachment is trained by a cross entropy over the candidates + 'no attachment' (target: the candidate node nearest to the GT end of the keypoint, within ATT_TGT m)."""''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
