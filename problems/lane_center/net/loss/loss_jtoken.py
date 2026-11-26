"""Losses of the token head (wiring likelihood, distillation, position head): the loss side of net/model/jtoken.py (D152; the functions are unchanged, `jtoken` holds the network)."""
import math
import torch
import torch.nn.functional as F

from jtoken import NEG, POS_CAP, SHORT, _const  # noqa: E402,I001


POS_LOSS = ("sum", "max")                                                   # the likelihood of a source with GT successors (`_source_nll`): -log sum over its group (default) or -log max (E376)


GEOM_KEYS = ("gt_pos", "gt_dir", "gt_btok", "gl")                           # E395: the GT geometry of a frame (GT nodes, their best tokens, the GT links with their ghost nodes), read by the curve / node loss only, never by the forward


def _frame_mean(x, m, w=None):
    """per frame the mean of x (B, N) over the entries of m, then the mean over the frames that have one (0 when none); w (B, N): the weight of every entry, the mean is then sum(w x) / sum(w) as torch's class-weighted cross-entropy
    (x is the already weighted term); w None = the plain mean"""
    per = (x * m).sum(1) / (m.sum(1).clamp_min(1) if w is None else (w * m).sum(1).clamp_min(1e-12))
    has = m.any(1)
    return per[has].mean() if has.any() else x.sum() * 0.0


# ------------------------------------------------------------------------------------------------------------------------------ loss
def _source_nll(lp, G, none_weight, pos_loss="sum"):
    """per source and class: lp (B, 2, K, K + 1) the log-probabilities of the per-source softmax (the none slot last), G (B, 2, K, K) the in-class GT successor groups -> (B, 2, K). G(i) non-empty: the likelihood of
    its GT successors, -log sum_{G(i)} p ('sum') or -log max_{G(i)} p ('max', E376: the mass on ONE member of the group; with one member the two are the same); G(i) empty: `none_weight` x -log p(none|i) (the sources
    with a GT successor are not touched by the weight)"""
    assert pos_loss in POS_LOSS, pos_loss
    K = G.shape[-1]
    g = lp[..., :K].masked_fill(~G, NEG)
    pos = -(torch.logsumexp(g, -1) if pos_loss == "sum" else g.amax(-1))
    return torch.where(G.any(-1), pos, -lp[..., K] * none_weight)


def source_rows(out):
    """(B, 2, K) bool: the sources of a class, the valid tokens with a candidate of it (the rows of the likelihood and of the KD term)"""
    return out["cand"].any(-1) & out["valid"][:, None]


def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5, none_weight=1.0, pos_loss="sum", wire_w=0.0, curve_w=0.0, node_w=0.0):
    """the likelihood of the module docstring -> (loss, info); batch: y (B, K, K), ylong (B, K, K), yscale (B, K, K, 2), ytype (B, K). wire_w (E392, default 0 = nothing added, `logp_rounds` not read): + wire_w x (nll_s + nll_l) of every early read
    of `out["logp_rounds"]` (the same `_source_nll`, none weight, pos_loss, rows and per-frame normalisation as the final one), in info nll1_s, nll1_l, nll2_s, nll2_l. info: nll_s, nll_l, ce1, ce3, scale_l1 and the counts gt_links, pos_pairs (in-gate GT
    links), gt_outside (links with no gate), gt_class (in the gate, but `ylong` disagrees with the token chord: a statistic, the class of a link is the token chord's). The long scale target is the fitted yscale
    of the long-class links (finite where the cache fitted one). curve_w / node_w (E395, default 0 = nothing read, nothing added, the value and the info of before): + curve_w x the curve loss and + node_w x the node loss of `curve_losses` on `out["dpos"]`
    (a head with pos_head) and the GT geometry keys `GEOM_KEYS` of the batch; info gets curve, node, n_curve, dpos_abs, dpos_cap. none_weight W (default 1: the loss of before): multiplies (i) the loss of the sources whose G(i) is empty (their target is the none slot), in both classes, and (ii) the type
    cross-entropy of the tokens whose type is none (class 0 weight W, the six others 1, normalised by the sum of the weights of the valid tokens of the frame, as torch's weighted mean). pos_loss 'sum' (default: the loss of
    before) or 'max' (E376): the likelihood of the sources with a non-empty G(i), in both classes (`_source_nll`); the empty-G terms, the type cross-entropy and the scale term are the same under both.
    scale_w = 0: no scale term, `yscale` is not read"""
    dev, cand, lp, valid = out["logp"].device, out["cand"], out["logp"], out["valid"]
    y, yl = batch["y"].to(dev) > 0, batch["ylong"].to(dev).bool()
    G = y[:, None] & cand                                                  # (B, 2, K, K): the in-gate GT successors of each class (the class of a pair = its token chord)
    yt = batch["ytype"].to(dev).long()
    nll = _source_nll(lp, G, none_weight, pos_loss)
    rows = source_rows(out)
    nll_s, nll_l = (_frame_mean(nll[:, c], rows[:, c]) for c in range(2))
    wt = torch.where(yt == 0, none_weight, 1.0) if none_weight != 1.0 else None
    ce1, ce3 = (_frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt) * (1.0 if wt is None else wt), valid, wt) for t in (out["types1"], out["types"]))
    loss = nll_s + nll_l + type_w * (ce1 + ce3)
    info = dict(nll_s=float(nll_s), nll_l=float(nll_l), ce1=float(ce1), ce3=float(ce3), gt_links=float(y.sum()), pos_pairs=float(G.sum()), gt_outside=float((y & ~cand.any(1)).sum()),
                gt_class=float((y & cand.any(1) & (yl != (out["chord"] > SHORT))).sum()))
    if wire_w:                                                              # E392: the same likelihood on every early read
        assert "logp_rounds" in out, "wire_w > 0 needs a head with rounds, in training mode for msg 'none' (the output has no early read)"
        for k, lpk in enumerate(out["logp_rounds"], 1):
            nk = _source_nll(lpk, G, none_weight, pos_loss)
            s_k, l_k = (_frame_mean(nk[:, c], rows[:, c]) for c in range(2))
            loss = loss + wire_w * (s_k + l_k)
            info.update({f"nll{k}_s": float(s_k), f"nll{k}_l": float(l_k)})
    if curve_w or node_w:                                                   # E395: the position head's loss on the curve through the refined nodes
        assert "dpos" in out and all(k in batch for k in GEOM_KEYS), f"curve_w / node_w need a head with pos_head (out['dpos']) and the GT geometry keys {GEOM_KEYS} in the batch (loader geom=True)"
        curve, node, ginfo = curve_losses(out["dpos"], batch["pos"].to(dev), batch["dirv"].to(dev), {k: batch[k].to(dev) for k in GEOM_KEYS}, valid=valid)
        loss = loss + (curve_w * curve if curve_w else 0.0) + (node_w * node if node_w else 0.0)
        info.update(ginfo)
    if not scale_w:
        return loss, info
    ys = batch["yscale"].to(dev).float()
    m = G[:, 1] & torch.isfinite(ys[..., 0])
    if m.any():
        l1 = (out["scale"].float()[m] - ys[m]).abs().mean()
        loss = loss + scale_w * l1
        info["scale_l1"] = float(l1)
    return loss, info


def teacher_logp(lps):
    """the log of the MEAN of the teachers' probabilities (not of their logits): lps = their log-probabilities (..., C) -> (..., C); one teacher -> its own bit for bit; a key that every teacher masks (log p -inf or the
    NEG fill) stays at p 0"""
    return torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))


def _kl(lpt, lp):
    """KL(p_T || p) per row from log-probabilities: the sum over the last axis of p_T (log p_T - log p); an entry with p_T = 0 (a masked key) adds exactly 0 (no 0 x log 0, no 0 x -inf)"""
    pt = lpt.exp()
    return torch.where(pt > 0, pt * (lpt - lp), 0.0).sum(-1)


def distill_loss(out, outs_t):
    """E379 (module docstring): out = the student's forward, outs_t = the frozen teachers' forwards on the SAME batch (no gradient) -> (kd, info: kd, its parts kd_s, kd_l, and ent_s, ent_l = the entropy of p_T per source
    of the class). The trainer adds W x kd to the likelihood and checks at its first step that the teachers saw the student's tokens (the same cand and valid)"""
    assert outs_t, "no teacher"
    lpt = teacher_logp([o["logp"] for o in outs_t])                        # (B, 2, K, K + 1): the candidates + the none slot
    src = source_rows(out)
    kl, ent = _kl(lpt, out["logp"]), -_kl(lpt, torch.zeros_like(lpt))      # H(p_T) = -KL(p_T || 1)
    kd_s, kd_l = (_frame_mean(kl[:, c], src[:, c]) for c in range(2))
    kd = kd_s + kd_l
    return kd, dict(kd=float(kd), kd_s=float(kd_s), kd_l=float(kd_l), ent_s=float(_frame_mean(ent[:, 0], src[:, 0])), ent_l=float(_frame_mean(ent[:, 1], src[:, 1])))


# ------------------------------------------------------------------------------------------------------------------------------ E395: the position head's curve loss
def cr_points_t(P0, P1, P2, P3, t):
    """the uniform Catmull-Rom segment from P1 (t = 0) to P2 (t = 1) with the ghost nodes P0 (before P1) and P3 (after P2): P* (..., 2), t (T,) -> (..., T, 2)"""
    t, c = t.to(P1.dtype)[:, None], lambda x: x[..., None, :]  # noqa: E731
    return 0.5 * (c(2 * P1) + c(-P0 + P2) * t + c(2 * P0 - 5 * P1 + 4 * P2 - P3) * t ** 2 + c(-P0 + 3 * P1 - 3 * P2 + P3) * t ** 3)


def cr_tangent_t(P0, P1, P2, P3, t):
    """d/dt of `cr_points_t` (same shapes)"""
    t, c = t.to(P1.dtype)[:, None], lambda x: x[..., None, :]  # noqa: E731
    return 0.5 * (c(-P0 + P2) + 2 * c(2 * P0 - 5 * P1 + 4 * P2 - P3) * t + 3 * c(-P0 + 3 * P1 - 3 * P2 + P3) * t ** 2)


def _rot90(v):
    return torch.stack([-v[..., 1], v[..., 0]], -1)


def _unit(v, eps):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _rho(x, beta):
    """smooth L1 with the knee beta: 0.5 x^2 / beta below it, |x| - 0.5 beta above"""
    a = x.abs()
    return torch.where(a < beta, 0.5 * x * x / beta, a - 0.5 * beta)


def refined_pos(pos, dirv, dpos):
    """(B, K, 2): pos + dpos x n, n = rot90(dirv) / max(|dirv|, 1e-9) the token's left normal. pos and dirv are the head's INPUTS: cut from the graph here (a gradient reaches dpos only)"""
    pos, dirv = _const(pos), _const(dirv)
    return pos + dpos[..., None] * _unit(_rot90(dirv), 1e-9)


def _mean_frames(x, m, ref):
    """x (B, N) over the entries of m: per frame the mean, then the mean over the frames that have one entry; no entry anywhere: ref.sum() x 0 (a zero with a graph, never NaN)"""
    return _frame_mean(torch.where(m, x, torch.zeros_like(x)), m) if m.any() else ref.sum() * 0.0


def curve_losses(dpos, pos, dirv, g, beta=0.5, ts=(0.0, 0.25, 0.5, 0.75, 1.0), valid=None, chord_max=8.0, ratio_max=2.0):
    """E395: the position head's losses -> (curve, node, info). dpos (B, K) the offsets, pos / dirv (B, K, 2) the tokens' inputs (`refined_pos`), g = the GT geometry of the batch (`GEOM_KEYS`): gt_pos / gt_dir (B, Ng, 2), gt_btok (B, Ng) long = the token index
    of the best token of a GT node (-1 = none, padding), gl (B, Lg, 4) long = (u, v, pre, suc) the GT link u -> v and the GT node before u / after v (-1 = none; u = -1: padding). Q[g] = the refined position of the best token of GT node g (a node without one is missing).
    node: per present node rho((Q - G) . n_g), n_g = rot90(gt_dir) normalised, beta the knee; per frame the mean over the nodes (no chord rule). curve: per link with Q[u], Q[v] present: the Catmull-Rom curve through (Q[pre], Q[u], Q[v], Q[suc]) against the one through
    the GT nodes, at the parameters ts, error ((C_pred - C_gt) . n_gt(t)) with n_gt(t) the unit normal of the GT curve (a sample with a GT tangent shorter than 1e-6 is dropped); a ghost is AVAILABLE when its index is >= 0 and its node is present, an
    unavailable ghost is the reflection (P0 = 2 P1 - P2, P3 = 2 P2 - P1) in BOTH curves; per frame the mean over the link x t samples. Both: then the mean over the frames that have one (none: dpos.sum() x 0). valid (B, K) bool = the tokens of
    the dpos statistics (default all). Exclusion by chord (the uniform Catmull-Rom curve overshoots with unequal chords): a link enters the curve loss only when, over the GT chords c0 = |G_u - G_pre| (an available ghost), c1 = |G_v - G_u| and
    c2 = |G_suc - G_v| (an available ghost), c1 <= chord_max and max(c) / min(c) <= ratio_max; the others are left out and counted. What the curve loss is: at t = 0 and t = 1 the node loss (Q - G) . n_gt (of node u / v, on the normal of the GT curve), at the other
    t a fixed combination, with weights summing to 1, of the lateral errors of four nodes (the curve is linear in the nodes), so in the quadratic regime it is a positive semi-definite quadratic form in the node errors; the target is the Catmull-Rom
    curve through the GT nodes, not the dense lane. info: curve, node, n_curve (the link x t samples used), n_gl (the GT links of the batch, u >= 0), n_links (those with both nodes present), n_excl (of them left out by the chord rule), dpos_abs (mean |dpos| of the
    valid tokens), dpos_cap (their share with |dpos| >= 0.99 POS_CAP)"""
    gp, gd, bt, gl = (g[k] for k in GEOM_KEYS)
    take = lambda a, i: torch.gather(a, 1, i.clamp_min(0)[..., None].expand(-1, -1, a.shape[-1]))  # noqa: E731   (B, N, C) at (B, L) indices
    Qg, present = take(refined_pos(pos, dirv, dpos), bt), bt >= 0                                              # (B, Ng, 2) the best tokens' refined positions
    has = lambda i: (i >= 0) & present.gather(1, i.clamp_min(0))  # noqa: E731   (B, L): the node i exists and has a token
    e_node = ((Qg - gp) * _unit(_rot90(gd), 1e-9)).sum(-1)
    node = _mean_frames(_rho(e_node, beta), present, dpos)
    u, v, pre, suc = gl.unbind(-1)
    ok, a_pre, a_suc = has(u) & has(v), has(pre), has(suc)

    def four(a, ap, asu):                                                   # (B, Ng, 2) -> the four nodes of every link, a ghost that is not available (ap / asu) reflected
        P1, P2 = take(a, u), take(a, v)
        return torch.where(ap[..., None], take(a, pre), 2 * P1 - P2), P1, P2, torch.where(asu[..., None], take(a, suc), 2 * P2 - P1)
    tt = torch.tensor(ts, dtype=Qg.dtype, device=Qg.device)
    G4 = four(gp, a_pre, a_suc)                                             # the same availability in both curves
    tan = cr_tangent_t(*G4, tt)                                             # (B, Lg, T, 2) the GT curve's tangent
    n_gt = _rot90(tan) / tan.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    e_cur = ((cr_points_t(*four(Qg, a_pre, a_suc), tt) - cr_points_t(*G4, tt)) * n_gt).sum(-1)
    ch = torch.stack([(G4[i + 1] - G4[i]).norm(dim=-1) for i in range(3)], -1)      # the GT chords c0, c1, c2 (B, Lg, 3); an unavailable ghost is the reflection, whose chord is c1 again: it changes neither the max nor the min
    use = ok & (ch[..., 1] <= chord_max) & (ch.amax(-1) <= ratio_max * ch.amin(-1))
    m = use[..., None] & (tan.norm(dim=-1) >= 1e-6)
    curve = _mean_frames(_rho(e_cur, beta).flatten(1), m.flatten(1), dpos)
    vt = torch.ones_like(dpos, dtype=torch.bool) if valid is None else valid.bool()
    ad, nv = dpos.detach().abs(), vt.sum().clamp_min(1)
    return curve, node, dict(curve=float(curve), node=float(node), n_curve=float(m.sum()), n_gl=float((u >= 0).sum()), n_links=float(ok.sum()), n_excl=float((ok & ~use).sum()), dpos_abs=float((ad * vt).sum() / nv),
                             dpos_cap=float(((ad >= 0.99 * POS_CAP) & vt).sum() / nv))
