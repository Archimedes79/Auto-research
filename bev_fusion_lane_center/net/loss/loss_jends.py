"""Losses of the junction pair head (pair logits, geometry, crossings): the loss side of net/model/jends.py (D152; the functions are unchanged, `jends` holds the network)."""
import math
import torch
import torch.nn.functional as F

from jends import ATT_TGT  # noqa: E402,I001
import qendnet as QEN


def _bezier_s(a, ta, b, tb, sa=None, sb=None, n=16):
    """points (..., n, 2) of the DELIVERED connector, the cubic Bezier of `qends.curve_s` in torch: control points a + sa L ta and b - sb L tb with L = the chord and the handle scales sa, sb (the E303 scale head; None = 1/3, which is `_hermite`'s chord-tangent curve);
    differentiable in the end points, the headings and the scales"""
    L = (b - a).norm(dim=-1, keepdim=True)
    sa_ = 1.0 / 3.0 if sa is None else sa[..., None]
    sb_ = 1.0 / 3.0 if sb is None else sb[..., None]
    s = torch.linspace(0, 1, n, device=a.device).view(*([1] * (a.dim() - 1)), n, 1)
    a_, b_ = a[..., None, :], b[..., None, :]
    q1, q2 = (a + sa_ * L * ta)[..., None, :], (b - sb_ * L * tb)[..., None, :]
    return (1 - s) ** 3 * a_ + 3 * (1 - s) ** 2 * s * q1 + 3 * (1 - s) * s ** 2 * q2 + s ** 3 * b_


def _dist_to_polyline(pts, poly):
    """(..., Np) distance of every point of `pts` (..., Np, 2) to the polyline `poly` (..., Nq, 2) (point-to-segment, so the sampling of the polyline does not enter); a small floor under the square root keeps the gradient finite on the line"""
    a, b = poly[..., :-1, :], poly[..., 1:, :]
    ab = b - a
    ap = pts[..., :, None, :] - a[..., None, :, :]
    t = ((ap * ab[..., None, :, :]).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)[..., None, :]).clamp(0.0, 1.0)
    d2 = ((ap - t[..., None] * ab[..., None, :, :]) ** 2).sum(-1)
    return (d2.min(-1)[0] + 1e-8).sqrt()


def _huber(d, delta=1.0):
    """smooth L1 of a distance: quadratic inside `delta` metres, linear beyond"""
    return torch.where(d < delta, 0.5 * d * d / delta, d - 0.5 * delta)


def crossing_penalty(out):
    """the soft order penalty: sum over (i, j), (i', j') of P(i -> j) P(i' -> j') where i, i' share an entry group, j, j' share an exit group, and the ranks cross (rank i < rank i' but rank j > rank j');
    P = sigmoid(logit) on the allowed pairs; normalised by the mass of P"""
    z = out["logit"].float()
    Pm = z.sigmoid() * out["allowed"].float()
    ga, ra, gb, rb = out["ga"], out["ra"], out["gb"], out["rb"]
    Rin = ((ga[:, :, None] == ga[:, None, :]) & (ra[:, :, None] < ra[:, None, :])).float()          # [i, i'] same entry group, rank i < rank i'
    Rout = ((gb[:, :, None] == gb[:, None, :]) & (rb[:, :, None] > rb[:, None, :])).float()         # [j, j'] same exit group, rank j > rank j'
    t = torch.bmm(torch.bmm(Rin, Pm), Rout.transpose(1, 2))                                          # (B, K, K): sum over i', j' of Rin[i, i'] P[i', j'] Rout[j, j']
    return (Pm * t).sum() / (Pm.sum() + 1.0)


def geom_loss(out, batch, y, allowed, ia, ib, bi, pv, pairs):
    """E317 / E318: the geometry of the delivered connector, supervised directly. (1) SHAPE: for every true pair whose two ends are keypoints (y = 1, allowed) the delivered curve (`_bezier_s`: the keypoint ends, the headings, the learned
    handle scales, 16 samples) against the GT connector polyline of the pair (`batch['ppts']`, 16 points of equal arc length): symmetric point-to-polyline distance, smooth L1 with delta 1 m, curve -> GT and GT -> curve. (2) ENDS: every valid keypoint
    within 4 m / 30 degrees of a GT end of its kind against the nearest such end, smooth L1 (E317) or, when the head has a position scale (`kp_logsig`, E318), the Laplace NLL |d| exp(-s) + s (the net may say that an end is uncertain, at the price of s).
    The gradient reaches the keypoint positions (through the soft-argmax into the heat, so into the maps head with --e2e), the headings (the dir maps) and the scales. Returns (loss or None, info); `ia`, `ib`: the matched GT end of every keypoint
    (E / X = none) as in `jends_loss` (they pick the GT polyline of a keypoint pair).
    E320 (`att_pos` in out): the curve and the end term use the ATTACHED ends (`Attach`), and the attachment is trained by a cross entropy over the candidates + 'no attachment' (target: the candidate node nearest to the GT end of the keypoint, within ATT_TGT m)."""
    dev = out["heat"].device
    B, K_ = out["valid"].shape[0], out["idx"].shape[2]
    E, X = batch["ent"].shape[1], batch["ext"].shape[1]
    pp = batch["ppts"].to(dev).float()
    PP = torch.full((B, E + 1, X + 1) + tuple(pp.shape[2:]), float("nan"), device=dev)
    PP[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = pp[pv]
    PPt = PP[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]                      # (B, K, K, N, 2): the GT polyline of the pair (keypoint i, keypoint j), NaN when it is no true pair
    m = (y > 0) & allowed & torch.isfinite(PPt[..., 0, 0])
    kp, kd = out["kp_pos"].float(), out["kp_dir"].float()
    ep, ed = (out["att_pos"].float(), out["att_dir"].float()) if "att_pos" in out else (kp, kd)               # E320: the FINAL ends of the delivered curve are the attached ones
    info, loss = {}, None
    if m.any():
        b_, i_, j_ = m.nonzero(as_tuple=True)
        sc = out.get("scale")
        curve = _bezier_s(ep[b_, i_], ed[b_, i_], ep[b_, K_ + j_], ed[b_, K_ + j_], None if sc is None else sc[b_, i_, j_, 0].float(), None if sc is None else sc[b_, i_, j_, 1].float())
        gt = PPt[m]
        d1, d2 = _dist_to_polyline(curve, gt), _dist_to_polyline(gt, curve)
        shape = _huber(d1).mean() + _huber(d2).mean()
        loss = shape
        info.update(geom_shape=float(shape), geom_n=float(m.sum()), geom_d_curve=float(d1.mean()), geom_d_gt=float(d2.mean()))
    ends, n_end, d_all = kp.new_zeros(()), 0, []
    ce_tot, st = kp.new_zeros(()), dict(n=0, ce=0.0, acc=0, near=0, bin=0, kp_d=[], att_d=[])                    # st: the attachment statistics over the keypoints with at least one candidate (critic 2026-11-12: the trivial dustbin cases and the keypoints without a node in reach are no information)
    for k, (key, mkey) in enumerate((("ent", "ent_m"), ("ext", "ext_m"))):
        G, gm = batch[key].to(dev).float(), batch[mkey].to(dev)
        kpk, kdk = kp[:, k * K_:(k + 1) * K_], kd[:, k * K_:(k + 1) * K_]
        with torch.no_grad():                                                                                # every valid keypoint of the kind is pulled to the nearest GT end within 4 m / 30 degrees (many to one: a real peak that
            dk = (kpk[:, :, None, :] - G[:, None, :, :2]).norm(dim=-1)                                        # lies next to the keypoint the BCE matched is a delivered connector end too, so its position counts)
            ang = (kdk[:, :, None, :] * G[:, None, :, 2:]).sum(-1)
            okm = (dk <= QEN.MATCH_POS) & (ang >= math.cos(math.radians(QEN.MATCH_ANG))) & gm[:, None, :] & out["valid"][:, k, :, None]
            a_ = torch.where(okm, dk, torch.full_like(dk, 1e6)).argmin(2)
            ok = okm.any(2)
        if not ok.any():
            continue
        g = G[..., :2].gather(1, a_[..., None].expand(-1, -1, 2))
        epk = ep[:, k * K_:(k + 1) * K_]
        d = ((epk - g) ** 2).sum(-1).add(1e-8).sqrt()[ok]
        if "att_logit" in out:                                                                               # E320: the attachment target of a keypoint = the node of B's head nearest to the GT end it belongs to (within ATT_TGT m), else 'no attachment'
            lg = out["att_logit"][:, k * K_:(k + 1) * K_]
            with torch.no_grad():
                dc = ((out["att_cpos"][:, k * K_:(k + 1) * K_] - g[:, :, None, :]) ** 2).sum(-1).sqrt()
                dc = torch.where(out["att_cvalid"][:, k * K_:(k + 1) * K_], dc, torch.full_like(dc, 1e6))
                tgt = torch.where(dc.min(2)[0] <= ATT_TGT, dc.argmin(2) + 1, torch.zeros_like(dc[..., 0], dtype=torch.long))
            ce_tot = ce_tot + F.cross_entropy(lg[ok], tgt[ok], reduction="sum")
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
        if "kp_logsig" in out:
            s = out["kp_logsig"][:, k * K_:(k + 1) * K_][ok]
            ends = ends + (d * torch.exp(-s) + s).sum()
            info["geom_sig"] = float(s.exp().mean())
        else:
            ends = ends + _huber(d).sum()
        n_end += int(ok.sum())
        d_all.append(d.detach())
    if n_end:
        ends = ends / n_end
        loss = ends if loss is None else loss + ends
        info.update(geom_end=float(ends), geom_end_d=float(torch.cat(d_all).mean()))
        if "att_logit" in out:                                                                               # E320: the cross entropy of the attachment over the candidates (+ the dustbin), normalised like the end term
            loss = loss + ce_tot / n_end
            if st["n"]:                                                                                      # logged over the keypoints with >= 1 candidate: att_acc against att_acc_nearest (= take the nearest candidate), att_ce (chance is ln(n_candidates + 1)), the attached against the keypoint end error
                info.update(att_n=float(st["n"]), att_ce=st["ce"] / st["n"], att_acc=st["acc"] / st["n"], att_acc_nearest=st["near"] / st["n"], att_bin=st["bin"] / st["n"],
                            att_kp_d=float(torch.cat(st["kp_d"]).mean()), att_d=float(torch.cat(st["att_d"]).mean()))
    return loss, info


def jends_loss(out, batch, cross_w=1.0, driven_w=0.0, scale_w=0.5, geom_w=0.0):
    dev = out["heat"].device
    B = out["heat"].shape[0]
    match = QEN.match_keypoints(out, batch)
    pairs = batch["pairs"].to(dev)
    E, X = batch["ent"].shape[1], batch["ext"].shape[1]
    T = torch.zeros(B, E + 1, X + 1, dtype=torch.bool, device=dev)
    pv = (pairs[..., 0] >= 0) & (pairs[..., 1] >= 0)
    bi = torch.arange(B, device=dev)[:, None].expand_as(pv)
    T[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = True
    ia, ib = match[0].clone(), match[1].clone()
    ia[ia < 0], ib[ib < 0] = E, X
    y = T[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]].float()
    allowed = out["allowed"]
    z = out["logit"].float()
    info = {}
    wmat = torch.ones_like(y)
    if driven_w > 0 and "pdrv" in batch:                                                                   # E298: the positive terms of the pairs the ego drives count 1 + driven_w times
        Dv = torch.zeros(B, E + 1, X + 1, device=dev)
        pd = batch["pdrv"].to(dev)
        Dv[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = pd[pv]
        yd = Dv[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]
        wmat = 1.0 + driven_w * yd * y
    if allowed.any():
        bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none", pos_weight=torch.tensor(3.0, device=dev)) * wmat * allowed).sum() / allowed.sum()
        dual = QEN._dual_softmax(z, y, allowed)
    else:
        bce = dual = z.sum() * 0.0
    loss = bce + 0.5 * dual
    info.update(bce=float(bce), pos_pairs=float((y * allowed).sum()), allowed=float(allowed.sum()))
    info["dual"] = float(dual)
    if "scale" in out and "pscale" in batch and scale_w > 0:                                                    # E303: L1 of the predicted handle scales against the fitted ones, on the matched true pairs
        S = torch.full((B, E + 1, X + 1, 2), float("nan"), device=dev)
        ps = batch["pscale"].to(dev)
        S[bi[pv], pairs[..., 0][pv], pairs[..., 1][pv]] = ps[pv]
        St = S[torch.arange(B, device=dev)[:, None, None], ia[:, :, None], ib[:, None, :]]
        m = (y > 0) & allowed & torch.isfinite(St[..., 0])
        if m.any():
            ls = (out["scale"][m] - St[m]).abs().mean()
            loss = loss + scale_w * ls
            info["scale_l1"] = float(ls)
    if "ga" in out and cross_w > 0 and allowed.any():
        cp = crossing_penalty(out)
        loss = loss + cross_w * cp
        info["cross"] = float(cp)
    if geom_w > 0:                                                                                              # E317 / E318 / E320: the shape and the ends of the delivered connector
        assert "ppts" in batch, "geom_w > 0 needs the GT connector polylines in the batch (`QData.DEFAULT_PPTS = True`, set by jends_train.py --geom-w): without them the whole geometry and attachment loss would be skipped silently"
        lg, ig = geom_loss(out, batch, y, allowed, ia, ib, bi, pv, pairs)
        if lg is not None:
            loss = loss + geom_w * lg
            info.update(ig)
    return loss, info
