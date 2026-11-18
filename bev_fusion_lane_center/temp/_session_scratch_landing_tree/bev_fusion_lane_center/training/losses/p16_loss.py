"""Loop 16: matching-free losses of the builder (deep supervision on every level of the tree).
Targets come from the exact distance / direction fields of the ground truth, the dense ground-truth points and the directed geodesic distance matrix (p16_data.py):
  verification   v_k of a sample on a ray = 1 iff the sample lies within 0.6 m of a lane AND the lane runs along the ray (so a survival product is a chord-validity check)
  child position distance of the child to the DOWNSTREAM ground-truth set of its parent (no assignment of children to ground-truth points)
  child validity 1 iff the child is within 0.8 m of that set and the chord parent -> child stays within 0.6 m of a lane
  graph          precision (edge samples to lane points) and recall (lane points to the nearest valid edge, soft minimum with a validity price), as in E120
  duplicates     validity-weighted repulsion of nodes closer than 1.5 m (geo) or a learned pair head (learned)
Nothing here uses argmax / Hungarian assignment of predictions; masks are built from labels and detached positions."""
import math
import os

import torch
import torch.nn.functional as F

import p16_builder as PB
import p16_geom as PG

UNREACH = 65535


def sample_gd(gd, q):
    """exact distance of the lane field (metres) at points q [B,...,2]"""
    return PB.samp(gd[:, None], q)[..., 0]


def sample_dir(gdir, q):
    d = PB.samp(gdir.float(), q)
    return F.normalize(d, dim=-1, eps=1e-6) * (d.norm(dim=-1, keepdim=True) > 0.5)


def nearest(pts, g, gm):
    """nearest ground-truth point of pts [B,P,2]: distance [B,P], index [B,P] (g [B,Gm,2], mask gm [B,Gm])"""
    d = torch.cdist(pts.float(), g)
    d = d.masked_fill(gm[:, None, :] < 0.5, 1e9)
    return d.min(-1)


def bce_p(p, t, w=None, eps=1e-4):
    p = p.float().clamp(eps, 1 - eps)
    l = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    if w is None:
        return l.mean()
    return (l * w).sum() / w.sum().clamp(min=1.0)


def seed_losses(sd, b):
    gd, gdir, g05, t05, m05 = b["gd"], b["gdir"], b["g05"], b["t05"], b["m05"]
    cpos, pos, dv, val = sd["cpos"], sd["pos"], sd["dvec"], sd["val"]
    near = sample_gd(gd, cpos) < 3.5  # candidate within snapping range of a lane
    dist, ix = nearest(pos, g05, m05)
    l_pos = (F.smooth_l1_loss(dist.clamp(max=3.0), torch.zeros_like(dist), beta=0.25, reduction="none") * near).sum() / near.sum().clamp(min=1)
    tg = t05.gather(1, ix[..., None].expand(-1, -1, 2))
    cos = (F.normalize(dv, dim=-1) * tg).sum(-1)
    l_dir = ((1 - cos) * near).sum() / near.sum().clamp(min=1)
    l_val = F.binary_cross_entropy_with_logits(val, near.float(), pos_weight=torch.tensor(2.0, device=val.device))
    return l_pos, l_dir, l_val


def downstream_mask(a, flag, b):
    """a [B,N] index of the parent's nearest 1 m point (+ flag): mask over the dense points [B,N,Gm] of the ground truth reachable downstream (flag > 0) or upstream (flag < 0) within 40 m"""
    dg = b["dg"]  # [B,Hm,Hm] int32 distance a -> x in 0.1 m
    row = dg.gather(1, a[..., None].expand(-1, -1, dg.shape[2]))  # [B,N,Hm] distance from a to x
    col = dg.transpose(1, 2).gather(1, a[..., None].expand(-1, -1, dg.shape[2]))  # distance from x to a
    r = torch.where((flag > 0)[..., None], row, col)
    reach = (r > 0) & (r <= 400)
    return reach.gather(2, b["n05"][:, None, :].expand(-1, a.shape[1], -1))


def vc_penalty(res, b, target=14.3, scale=2.5):
    """SPF1: expected number of DISTINCT vertices per 100 m of lane against the Douglas-Peucker target (soft count: every node counts pi / (local validity density at the vertex scale));
    penalty = relu(N_eff x 100 / L_gt - target) / target, averaged over the batch"""
    pos, pi, valid = res["pos"], res["pi"] * res["valid"], res["valid"]
    d = torch.cdist(pos.float(), pos.float())
    dens = (torch.exp(-(d ** 2) / (2 * scale ** 2)) * pi[:, None, :]).sum(-1).clamp(min=1e-3)
    n_eff = (pi / dens).sum(-1)
    L = (b["m05"].sum(-1) * 0.5).clamp(min=20.0)  # lane length in the frame (m)
    dens100 = n_eff * 100.0 / L
    return F.relu(dens100 - target).div(target).mean(), float(dens100.mean())


def l0_penalty(res, b, target=14.3):
    """SPF2: expected number of open gates of the plausible nodes per 100 m (price of the hard-concrete gate)"""
    w = res["pi_tree"].detach() * res["valid"]
    L = (b["m05"].sum(-1) * 0.5).clamp(min=20.0)
    return ((res["p_open"] * w).sum(-1) * 100.0 / L / target).mean()


def level_losses(lv, b, pat, hazard=True, mcl=False, eps_mcl=0.1, dup_label=False, cens=False, fc=False):
    """verification BCE on all samples of the tips, child position / validity against the downstream set"""
    o = lv["out"]
    q, v_logit = o["q"], o["v_logit"]
    ray_d = o["ray_d"]
    B, N, R, K, _ = q.shape
    valid = lv["tips_valid"]
    gdq = sample_gd(b["gd"], q)
    dq = sample_dir(b["gdir"], q)
    sgn = lv["tips_flag"][:, :, None, None]  # forward tips grow along the travel direction, backward tips against it
    tgt = ((gdq < 0.6) & ((dq * ray_d[:, :, :, None, :]).sum(-1) * sgn > 0.5)).float()
    # tips that are not on a lane carry less weight (nothing can be verified there)
    ta_d, ta = nearest(lv["tips_pos"].detach(), b["g1"], b["m1"])
    on = (ta_d < 2.5).float()
    w = pat.mask[None, None] * valid[:, :, None, None] * (0.3 + 0.7 * on)[:, :, None, None]
    if cens:  # censoring (VS01): samples outside the raster (border padding) are not read, so they carry no label
        w = w * (q.abs().amax(-1) < 49.5).float()
    if hazard:  # discrete-time hazard training (VS finding, K02): a sample only counts while all earlier samples of its ray were verified; the first failure is the last counted one
        prev_ok = torch.cumprod(torch.cat([torch.ones_like(tgt[..., :1]), tgt[..., :-1]], -1), -1)
        w = w * prev_ok
    l_ver = (F.binary_cross_entropy_with_logits(v_logit, tgt, reduction="none") * w).sum() / w.sum().clamp(min=1)
    if "loi_logit" in o:  # baseline B3: one label per ray (every sample of the ray verified)
        tray = (tgt + (1 - pat.mask[None, None])).prod(-1)
        wr = valid[:, :, None] * (0.3 + 0.7 * on)[:, :, None] * torch.ones_like(tray)
        l_ver = l_ver + (F.binary_cross_entropy_with_logits(o["loi_logit"], tray, reduction="none") * wr).sum() / wr.sum().clamp(min=1)
    # children
    end, nu, ok = lv["child_pos"], lv["child_nu"], lv["child_ok"]
    bch = end.shape[2]
    dmask = downstream_mask(ta, lv["tips_flag"], b) & (on > 0.5)[..., None]  # [B,N,Gm]
    cd = torch.cdist(end.reshape(B, N * bch, 2).float(), b["g05"]).reshape(B, N, bch, -1)
    cd = cd.masked_fill(~dmask[:, :, None, :], 1e9)
    dist = cd.min(-1).values  # distance to the downstream set  [B,N,b]
    chord_pts = lv["tips_pos"][:, :, None, None, :] + (end[:, :, :, None, :] - lv["tips_pos"][:, :, None, None, :]) * torch.tensor([0.2, 0.4, 0.6, 0.8], device=end.device)[None, None, None, :, None]
    chord_ok = (sample_gd(b["gd"], chord_pts.detach()) < 0.6).all(-1)
    tgt_c = ((dist < 0.8) & chord_ok).float()
    if dup_label and bch > 1:  # best-per-ground-truth duplicate label (MH finding): a child that lies on the chord of a longer valid sibling is the same hypothesis twice -> negative
        tip = lv["tips_pos"]
        reach = (end - tip[:, :, None, :]).norm(dim=-1)  # [B,N,b]
        sdj = PG.seg_dist(end[:, :, :, None, :], tip[:, :, None, None, :], end[:, :, None, :, :])  # [B,N,j,i]: child j against the chord tip -> child i
        longer = reach[:, :, None, :] > reach[:, :, :, None] + 1e-6
        dup = (sdj < 1.5) & longer & (tgt_c[:, :, None, :] > 0.5)
        tgt_c = tgt_c * (~dup.any(-1)).float()
    wm =(ok & (valid[:, :, None] > 0)).float()
    near_ = (dist < 2.5).float() * wm
    if mcl and bch > 1:  # multiple-choice learning: the child closest to the downstream set carries the position loss, the others a small share (relaxed winner-takes-all)
        best = (dist <= dist.min(-1, keepdim=True).values + 1e-6).float()
        near_ = near_ * (best + eps_mcl * (1 - best))
    l_pos = (F.smooth_l1_loss(dist.clamp(max=3.0), torch.zeros_like(dist), beta=0.25, reduction="none") * near_).sum() / near_.sum().clamp(min=1)
    l_val = bce_p(nu, tgt_c.detach(), wm)
    stats = dict(ver_pos=float((tgt * w).sum() / w.sum().clamp(min=1)), child_ok=float((tgt_c * wm).sum() / wm.sum().clamp(min=1)), child_near=float(near_.sum() / wm.sum().clamp(min=1)))
    l_cov = torch.zeros((), device=end.device)
    if fc and bch > 1:  # FC (fork coverage): a tip that stands up to 14 m before a ground-truth fork (merge for backward tips) must cover the ground-truth lane points 10 - 16 m downstream with its children, so that every branch is hypothesised (set coverage, no assignment)
        dgm = b["dg"]
        fwd = (lv["tips_flag"] > 0)[..., None]
        row = dgm.gather(1, ta[..., None].expand(-1, -1, dgm.shape[2]))  # [B,N,Hm] distance tip -> x (0.1 m)
        col = dgm.transpose(1, 2).gather(1, ta[..., None].expand(-1, -1, dgm.shape[2]))  # x -> tip
        r = torch.where(fwd, row, col)
        rg = r.gather(2, b["n05"][:, None, :].expand(-1, N, -1))  # [B,N,Gm]
        ring = (rg >= 100) & (rg <= 160) & (b["m05"][:, None, :] > 0.5)
        jl_f = torch.where(fwd[..., 0, None], torch.ones(1, device=r.device), torch.zeros(1, device=r.device))  # [B,N,1] 1 = forward tip
        fork_tip = torch.zeros(B, N, dtype=torch.bool, device=r.device)
        for key, is_fwd in (("js", True), ("jm", False)):
            lm = b[key]  # [B,J,2]
            if lm.shape[1] == 0:
                continue
            nl = torch.cdist(lm.float(), b["g1"]).argmin(-1)  # nearest 1 m lane point of every landmark [B,J]
            dl = r.gather(2, nl[:, None, :].expand(-1, N, -1))  # [B,N,J] geodesic tip -> landmark
            ok_l = (lm[:, None, :, 0] < 1e3) & (dl > 0) & (dl <= 140)
            fork_tip = fork_tip | (ok_l.any(-1) & (lv["tips_flag"] > 0 if is_fwd else lv["tips_flag"] < 0))
        use = fork_tip & (on > 0.5) & (valid > 0.5) & ring.any(-1)
        if use.any():
            ce = torch.cdist(end.reshape(B, N * bch, 2).float(), b["g05"]).reshape(B, N, bch, -1).min(2).values  # [B,N,Gm] nearest child end of every lane point
            slack = F.relu(ce - 1.2).clamp(max=3.0)
            per_tip = (F.smooth_l1_loss(slack, torch.zeros_like(slack), beta=0.25, reduction="none") * ring).sum(-1) / ring.sum(-1).clamp(min=1)
            l_cov = (per_tip * use).sum() / use.sum().clamp(min=1)
            stats["fork_tips"] = float(use.float().mean())
    return l_ver, l_pos, l_val, stats, l_cov


FAR_W = float(os.environ.get("P18_FAR_W", "0"))  # loop 18: range-weighted recall term (0 = off, identical to loop 16)
REC_CLAMP = float(os.environ.get("P19_REC_CLAMP", "3.0"))  # loop 19 (E244 arm b): clamp (m) of the soft minimum of the recall term; 3.0 = identical to loops 16-18, 8.0 removes the dead zone


def graph_losses(res, b, pos_key="pos"):
    """edge precision and soft recall over the whole tree (nodes with a parent); validity price in the recall term"""
    pos, par, pi, valid, flag = res[pos_key], res["parent"], res["pi"], res["valid"], res["flag"]
    B, Nn, _ = pos.shape
    has = (par >= 0) & (valid > 0)
    pp = pos.gather(1, par.clamp(min=0)[..., None].expand(-1, -1, 2))
    fr = torch.tensor([0.25, 0.5, 0.75, 1.0], device=pos.device)[None, None, :, None]
    Q = pp[:, :, None, :] + fr * (pos - pp)[:, :, None, :]  # [B,Nn,4,2]
    dq = torch.cdist(Q.reshape(B, -1, 2).float(), b["g05"]).masked_fill(b["m05"][:, None, :] < 0.5, 1e9).min(-1).values.reshape(B, Nn, 4)
    use = (dq.min(-1).values < 2.5).float() * has.float()  # edges that are near a lane at all
    l_prec = (F.smooth_l1_loss(dq.clamp(max=3.0), torch.zeros_like(dq), beta=0.25, reduction="none").mean(-1) * use).sum() / use.sum().clamp(min=1)
    # recall: every 2nd lane point (1 m) to the nearest edge, price 2 m x (1 - validity)
    gp = b["g05"][:, ::2]
    gm = b["m05"][:, ::2]
    a, c = pp, pos
    ab = c - a
    L2 = (ab * ab).sum(-1).clamp(min=1e-6)
    ga = gp[:, :, None, :] - a[:, None, :, :]
    t = ((ga * ab[:, None]).sum(-1) / L2[:, None]).clamp(0, 1)
    proj = a[:, None] + t[..., None] * ab[:, None]
    sd = (gp[:, :, None, :] - proj).norm(dim=-1)  # [B,G,Nn]
    price = 2.0 * (1.0 - pi.clamp(0, 1))
    sd = sd + price[:, None, :]
    sd = sd.masked_fill(~has[:, None, :], 1e9)
    dg_ = sd.min(-1).values.clamp(max=REC_CLAMP)
    gwt = gm * (gp.abs().max(-1).values < 48.0)
    if FAR_W > 0:  # loop 18: extra weight on ground-truth points beyond 25 m (smooth ramp to 40 m): the recall gap to the control lies there
        rg = gp.norm(dim=-1)
        gwt = gwt * (1.0 + FAR_W * ((rg - 25.0) / 15.0).clamp(0, 1))
    l_rec = (F.smooth_l1_loss(dg_, torch.zeros_like(dg_), beta=0.25, reduction="none") * gwt).sum() / gwt.sum().clamp(min=1)
    return l_prec, l_rec


PHD_SIGMA, PHD_GAMMA, PHD_MIX, PHD_EPS, PHD_RHO = 0.6, 2.0, 0.8, 1e-3, 1.0  # loop 19 (E244, K193): cross-track density 0.8 Gaussian (0.6 m) + 0.2 Cauchy (2 m), 1 point per m


def phd_loss(res, b, pos_key="pos"):
    """K193 (loop 19, E244): matching-free Poisson / PHD set likelihood of the ground-truth lane points under the intensity of the valid edges.
    lambda(x) = rho * sum_i pi_i k(d_i(x)) over edges (parent -> node) with k = 0.8 Gaussian(0.6 m) + 0.2 Cauchy(2 m) of the distance d_i to the segment;
    L = [ rho * sum_i pi_i l_i  -  sum_j log(lambda(g_j) + eps) ] / n_gt  with l_i the edge length, g_j every 2nd 0.5 m ground-truth point (1 per m).
    A log of a sum, not a minimum: every edge within about 10 m of a point gets a gradient (no clamp dead zone); pi becomes the supported fraction of the edge
    length, and two duplicate edges share their mass. No argmax, no matching; gradient reaches pi and the node positions."""
    pos, par, pi, valid = res[pos_key].float(), res["parent"], res["pi"].float(), res["valid"]
    has = (par >= 0) & (valid > 0)
    a = pos.gather(1, par.clamp(min=0)[..., None].expand(-1, -1, 2))
    ab = pos - a
    L2 = (ab * ab).sum(-1).clamp(min=1e-6)
    gp, gm = b["g05"][:, ::2].float(), b["m05"][:, ::2].float()
    w = gm * (gp.abs().max(-1).values < 48.0)
    t = (((gp[:, :, None, :] - a[:, None]) * ab[:, None]).sum(-1) / L2[:, None]).clamp(0, 1)
    d = (gp[:, :, None, :] - (a[:, None] + t[..., None] * ab[:, None])).norm(dim=-1)  # [B,G,Nn]
    k = PHD_MIX * torch.exp(-0.5 * (d / PHD_SIGMA) ** 2) / (PHD_SIGMA * math.sqrt(2 * math.pi)) + (1 - PHD_MIX) / (math.pi * PHD_GAMMA * (1 + (d / PHD_GAMMA) ** 2))
    pv = pi.clamp(0, 1) * has.float()
    lam = PHD_RHO * (k * pv[:, None, :]).sum(-1)  # [B,G]
    mass = PHD_RHO * (pv * L2.sqrt()).sum(-1)  # [B]
    n = w.sum(-1).clamp(min=1.0)
    return ((mass - (torch.log(lam + PHD_EPS) * w).sum(-1)) / n).mean()


def junc_loss(res, b, sig=2.5):
    """split / merge heat maps against Gaussians (sigma 2.5 m) at the ground-truth landmarks; weighted like the seed heat map (positives x 10)"""
    lg = res["junc_logit"].float()
    B, _, H, W = lg.shape
    ax = (torch.arange(H, device=lg.device, dtype=torch.float32) + 0.5) * 0.5 - 50.0
    g0, g1 = torch.meshgrid(ax, ax, indexing="ij")
    cells = torch.stack([g0, g1], -1).reshape(1, -1, 2)
    tg = []
    for key in ("js", "jm"):
        d2 = ((cells[:, :, None, :] - b[key][:, None, :, :]) ** 2).sum(-1)  # [B, cells, P]
        tg.append(torch.exp(-d2.min(-1).values / (2 * sig ** 2)).reshape(B, H, W))
    t = torch.stack(tg, 1)
    bce = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
    return (bce * (1.0 + 9.0 * t)).mean()


def dup_geo(res, r=1.5, sig=0.6):
    """validity-weighted repulsion of node pairs closer than r (duplicates); parent-child pairs are >= 2 m apart by construction"""
    pos, pi, valid = res["pos"], res["pi"], res["valid"]
    B, Nn, _ = pos.shape
    d = torch.cdist(pos.float(), pos.float())
    m = (d < r) & (valid[:, :, None] > 0) & (valid[:, None, :] > 0) & ~torch.eye(Nn, dtype=torch.bool, device=pos.device)[None]
    k = torch.exp(-(d ** 2) / (2 * sig ** 2)) * pi[:, :, None] * pi[:, None, :] * m
    return k.sum() / (2.0 * valid.sum().clamp(min=1))


def dup_learned(builder, res, b, k=8):
    """pair head over the k nearest valid nodes; target: both nodes within 1 m of a lane and within 1.5 m of each other along the lane graph"""
    pos, pi, valid, h, th = res["pos"].detach(), res["pi"], res["valid"], res["h"], res["th_travel"]
    B, Nn, _ = pos.shape
    d = torch.cdist(pos.float(), pos.float())
    d = d.masked_fill(~(valid[:, None, :] > 0) | torch.eye(Nn, dtype=torch.bool, device=pos.device)[None], 1e9)
    kd, ki = d.topk(k, dim=-1, largest=False)  # k nearest nodes  (neighbour list from positions)
    hp = builder.pair_proj(h.float())
    hj = PB.gather1(hp, ki.reshape(B, -1)).reshape(B, Nn, k, -1)
    rel = PB.gather1(pos, ki.reshape(B, -1)).reshape(B, Nn, k, 2) - pos[:, :, None, :]
    dth = PB.gather1(th, ki.reshape(B, -1)).reshape(B, Nn, k) - th[:, :, None]
    pj = PB.gather1(pi, ki.reshape(B, -1)).reshape(B, Nn, k)
    f = torch.cat([hp[:, :, None, :].expand(-1, -1, k, -1), hj, rel / 3.0, torch.cos(dth)[..., None], torch.sin(dth)[..., None], pi[:, :, None, None].expand(-1, -1, k, 1) - pj[..., None]], -1)
    logit = builder.pair(f)[..., 0]
    # labels
    da, a = nearest(pos, b["g1"], b["m1"])
    aj = PB.gather1(a, ki.reshape(B, -1)).reshape(B, Nn, k)
    daj = PB.gather1(da, ki.reshape(B, -1)).reshape(B, Nn, k)
    dg = b["dg"]
    d1 = dg.gather(1, a[..., None].expand(-1, -1, dg.shape[2])).gather(2, aj)  # a_i -> a_j
    d2 = dg.gather(1, aj.reshape(B, -1, 1).expand(-1, -1, dg.shape[2])).reshape(B, Nn, k, -1).gather(3, a[:, :, None, None].expand(-1, -1, k, 1))[..., 0]  # a_j -> a_i
    same = (torch.minimum(d1, d2) <= 15) | (aj == a[:, :, None])
    tgt = (same & (da[:, :, None] < 1.0) & (daj < 1.0) & (kd < 3.0)).float()
    w = ((kd < 3.0) & (valid[:, :, None] > 0)).float()
    l = (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none") * w).sum() / w.sum().clamp(min=1)
    return l, dict(logit=logit, ki=ki, kd=kd, tgt=tgt, w=w)


def link_loss(res, b):
    """L: BCE of the link logits; positive iff both ends lie within 1.2 m of a lane and the second end is 0 - 15 m DOWNSTREAM of the first along the ground-truth lane graph;
    plus the connector verification (samples between the two ends on a lane)"""
    off = res["link_off"]
    pos = res["pos"][:, off:]
    valid = res["valid"][:, off:]
    ki, logit, mask = res["link_ki"], res["link_logit"], res["link_mask"]
    B, M, k = ki.shape
    da, a = nearest(pos.detach(), b["g1"], b["m1"])
    aj = PB.gather1(a, ki.reshape(B, -1)).reshape(B, M, k)
    daj = PB.gather1(da, ki.reshape(B, -1)).reshape(B, M, k)
    dg = b["dg"]
    row = dg.gather(1, a[..., None].expand(-1, -1, dg.shape[2]))  # [B,M,Hm]
    dij = row.gather(2, aj)  # [B,M,k]
    tgt = ((da[:, :, None] < 1.2) & (daj < 1.2) & (dij > 0) & (dij <= 150)).float()
    w = mask.float() * (valid[:, :, None] > 0).float()
    l_link = (F.binary_cross_entropy_with_logits(logit, tgt, reduction="none", pos_weight=torch.tensor(2.0, device=logit.device)) * w).sum() / w.sum().clamp(min=1)
    return l_link, dict(link_pos=float((tgt * w).sum() / w.sum().clamp(min=1)))


DEFAULT_W = dict(fc=0.0, junc=1.0, cens=0.0, dupl=0.0, vc=0.0, l0=1.0, link=1.0, ver=1.0, pos=2.0, val=1.0, prec=1.0, rec=1.0, seed_pos=1.0, seed_dir=0.3, seed_val=0.5, dup=0.5, pair=0.5, rev=1.0, aux_seed=0.5, aux_dir=0.2, level_decay=0.8)


def total_loss(builder, res, b, w=None):
    w = {**DEFAULT_W, **(w or {})}
    pat = builder.pat
    parts, tot = {}, 0.0
    sp, sdir, sval = seed_losses(res["seeds"], b)
    tot = tot + w["seed_pos"] * sp + w["seed_dir"] * sdir + w["seed_val"] * sval
    parts.update(seed_pos=sp, seed_dir=sdir, seed_val=sval)
    lv_stats = {}
    for i, lv in enumerate(res["levels"]):
        lw = w["level_decay"] ** i if len(res["levels"]) > 1 else 1.0
        l_ver, l_pos, l_val, st, l_cov = level_losses(lv, b, pat, mcl=builder.cfg.mcl, dup_label=w["dupl"] > 0, cens=w["cens"] > 0, fc=w["fc"] > 0)
        tot = tot + lw * (w["ver"] * l_ver + w["pos"] * l_pos + w["val"] * l_val + w["fc"] * l_cov)
        for k_, v_ in (("ver", l_ver), ("pos", l_pos), ("val", l_val), ("cov", l_cov)):
            parts[k_] = parts.get(k_, 0.0) + float(v_) / len(res["levels"])
        for k_, v_ in st.items():
            lv_stats[k_] = lv_stats.get(k_, 0.0) + v_ / len(res["levels"])
    l_prec, l_rec = graph_losses(res, b)
    tot = tot + w["prec"] * l_prec + w["rec"] * l_rec
    parts.update(prec=l_prec, rec=l_rec)
    if w.get("phd", 0.0) > 0:  # loop 19 (E244 arm a): Poisson set likelihood of the lane points under the edge intensity; default off
        l_phd = phd_loss(res, b)
        tot = tot + w["phd"] * l_phd
        parts["phd"] = l_phd
    if w["vc"] > 0:
        l_vc, dens = vc_penalty(res, b)
        tot = tot + w["vc"] * l_vc
        parts.update(vc=l_vc, dens100=dens)
    if "p_open" in res:
        l_0 = l0_penalty(res, b)
        tot = tot + w["l0"] * l_0
        parts["l0"] = l_0
    if builder.cfg.dedup == "geo":
        l_dup = dup_geo(res)
        tot = tot + w["dup"] * l_dup
        parts["dup"] = l_dup
    else:
        l_pair, _ = dup_learned(builder, res, b)
        tot = tot + w["pair"] * l_pair
        parts["pair"] = l_pair
    if "link_ki" in res:
        l_link, st = link_loss(res, b)
        tot = tot + w["link"] * l_link
        parts["link"] = l_link
        lv_stats.update(st)
    if "junc_logit" in res:
        l_j = junc_loss(res, b)
        tot = tot + w["junc"] * l_j
        parts["junc"] = l_j
    if "pos_rev" in res:
        rp, rr = graph_losses(res, b, pos_key="pos_rev")
        kd, _ = nearest(res["pos_rev"].detach(), b["g05"], b["m05"])
        vm = res["valid"] * (res["pi"].detach() > 0.1).float()
        kt = (kd < 1.0).float()[None].expand(res["keep_rev"].shape[0], -1, -1)
        l_keep = (F.binary_cross_entropy_with_logits(res["keep_rev"].float(), kt, reduction="none") * vm[None]).sum() / (vm.sum() * kt.shape[0]).clamp(min=1)
        tot = tot + w["rev"] * (rp + rr) + 0.2 * l_keep
        parts["rev_prec"], parts["rev_rec"], parts["rev_keep"] = rp, rr, l_keep
    # keep the stem heads (seed heat map, direction field) trained as in E94
    import p10_train as T

    a1 = T.seed_loss(res["seed_logit"].float(), b["gd"])
    a2 = T.dir_loss(res["dirh"].float(), b["gdir"].float(), b["gd"])
    tot = tot + w["aux_seed"] * a1 + w["aux_dir"] * a2
    parts.update(aux_seed=a1, aux_dir=a2)
    out = {k: float(v) for k, v in parts.items()}
    out.update(lv_stats)
    out["tot"] = float(tot)
    return tot, out
