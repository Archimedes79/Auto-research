def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)

edit("temp/e320/dev/jends.pysrc", [
    ('''    ce_tot, n_acc, n_bin, kp_d = kp.new_zeros(()), 0, 0, []
''', '''    ce_tot, st = kp.new_zeros(()), dict(n=0, ce=0.0, acc=0, near=0, bin=0, kp_d=[], att_d=[])                    # st: the attachment statistics over the keypoints with at least one candidate (critic 2026-11-12: the trivial dustbin cases and the keypoints without a node in reach are no information)
'''),
    ('''            ce_tot = ce_tot + F.cross_entropy(lg[ok], tgt[ok], reduction="sum")
            n_acc += int((lg[ok].argmax(-1) == tgt[ok]).sum())
            n_bin += int((tgt[ok] == 0).sum())
            kp_d.append(((kpk - g) ** 2).sum(-1).add(1e-8).sqrt()[ok].detach())
''', '''            ce_tot = ce_tot + F.cross_entropy(lg[ok], tgt[ok], reduction="sum")
            with torch.no_grad():
                sel = ok & out["att_cvalid"][:, k * K_:(k + 1) * K_].any(-1)
                if sel.any():
                    st["n"] += int(sel.sum())
                    st["ce"] += float(F.cross_entropy(lg[sel], tgt[sel], reduction="sum"))
                    st["acc"] += int((lg[sel].argmax(-1) == tgt[sel]).sum())
                    st["near"] += int((tgt[sel] == 1).sum())                                                  # the baseline 'take the nearest candidate' (the candidates are sorted by distance)
                    st["bin"] += int((tgt[sel] == 0).sum())
                    st["kp_d"].append(((kpk - g) ** 2).sum(-1).add(1e-8).sqrt()[sel])
                    st["att_d"].append(((epk - g) ** 2).sum(-1).add(1e-8).sqrt()[sel])
'''),
    ('''        if kp_d:                                                                                             # E320: the cross entropy of the attachment over the candidates (+ the dustbin)
            ce = ce_tot / n_end
            loss = loss + ce
            info.update(att_ce=float(ce), att_acc=n_acc / n_end, att_bin=n_bin / n_end, att_kp_d=float(torch.cat(kp_d).mean()))''', '''        if "att_logit" in out:                                                                               # E320: the cross entropy of the attachment over the candidates (+ the dustbin), normalised like the end term
            loss = loss + ce_tot / n_end
            if st["n"]:                                                                                      # logged over the keypoints with >= 1 candidate: att_acc against att_acc_nearest (= take the nearest candidate), att_ce (chance is ln(n_candidates + 1)), the attached against the keypoint end error
                info.update(att_n=float(st["n"]), att_ce=st["ce"] / st["n"], att_acc=st["acc"] / st["n"], att_acc_nearest=st["near"] / st["n"], att_bin=st["bin"] / st["n"],
                            att_kp_d=float(torch.cat(st["kp_d"]).mean()), att_d=float(torch.cat(st["att_d"]).mean()))'''),
])
print("k3 ok")
