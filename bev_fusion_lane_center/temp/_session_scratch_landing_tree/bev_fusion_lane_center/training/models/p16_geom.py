"""Loop 16: torch geometry of the verified-ray hypothesis tree (pattern, survival read-out, hypothesis selection) - the batched twin of p16_ref.py (numpy reference).
Everything here is parameter-free. The functions marked SELECTION pick items (argmax index, top-b, NMS); the gradient flows through the picked items and their scores, never through the choice.
Shapes: tips [B,N], rays R, samples K, children b.   Coordinates: metres in the ego frame, heading th: u = (cos th, sin th)."""
import math

import torch

PHIS = (0, 9, -9, 20, -20, 50, -50, 90, -90, 125, -125, 180)
RINGS = (4.0, 8.0, 12.0, 18.0, 24.0)
CFG = dict(r_cap=24.0, r_floor=4.0, phi_c=20.0, prior_deg=45.0, prior_floor=0.02, cov_r=2.5, nms_seg=1.2, nms_r=2.0, alive=0.3, skip_penalty=0.5, v_floor=0.02, v_ceil=0.99)


def envelope(phi_deg, cfg=CFG):
    return max(cfg["r_floor"], cfg["r_cap"] / (1.0 + (phi_deg / cfg["phi_c"]) ** 2))


class Pattern:
    """ray pattern of a tip: phi [R] (rad), radii [R,K], mask [R,K], axis_deg [R] (|bearing|), prior [R]"""

    def __init__(self, kind="teardrop", cfg=CFG, device="cpu", reach=None):
        if reach is not None:  # SPF3: shorter or longer ray ladder (envelope r_cap = reach, rings up to the reach)
            cfg = dict(cfg, r_cap=float(reach))
        if kind == "teardrop":
            phis = [float(a) for a in PHIS]
            radii, axis = [], []
            for a in phis:
                ax = abs(a)
                r = envelope(ax, cfg)
                radii.append([rk for rk in RINGS if rk < r - 1e-6] + [r])
                axis.append(ax)
            prior = [max(cfg["prior_floor"], math.exp(-((ax / cfg["prior_deg"]) ** 2))) for ax in axis]
        elif kind == "fixed":  # E94-like fan: 9 headings, +-45 deg, samples at 2 / 4 / 6 m, no direction prior
            phis = [(i - 4) * 11.25 for i in range(9)]
            sc = (float(reach) / 6.0) if reach is not None else 1.0  # SPF3 for the fan: longer steps (reach 12 m = samples at 4 / 8 / 12 m)
            radii = [[2.0 * sc, 4.0 * sc, 6.0 * sc] for _ in phis]
            axis = [abs(a) for a in phis]
            prior = [1.0] * len(phis)
        else:
            raise ValueError(kind)
        self.kind, self.cfg = kind, cfg
        self.R, self.K = len(phis), max(len(r) for r in radii)
        rad = torch.zeros(self.R, self.K)
        msk = torch.zeros(self.R, self.K)
        for i, r in enumerate(radii):
            rad[i, : len(r)] = torch.tensor(r)
            msk[i, : len(r)] = 1.0
        self.phi = torch.deg2rad(torch.tensor(phis)).to(device)
        self.phi_deg = torch.tensor(phis).to(device)
        self.radii, self.mask = rad.to(device), msk.to(device)
        self.axis_deg = torch.tensor(axis).to(device)
        self.prior = torch.tensor(prior).to(device)
        self.n_samples = int(msk.sum().item())
        self.reach = (rad * msk).max(1).values.to(device)  # ray length

    def to(self, device):
        for k in ("phi", "phi_deg", "radii", "mask", "axis_deg", "prior", "reach"):
            setattr(self, k, getattr(self, k).to(device))
        return self


def ray_points(pos, th, pat):
    """sample points of all rays of all tips: pos [B,N,2], th [B,N] -> [B,N,R,K,2]"""
    ang = th[..., None] + pat.phi  # [B,N,R]
    d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)  # [B,N,R,2]
    return pos[:, :, None, None, :] + pat.radii[None, None, :, :, None] * d[:, :, :, None, :]


def survival(v, mask):
    """v [...,K] verification probabilities (masked samples count as 0). T_k = v_1 ... v_k; P_k = T_k (1 - v_(k+1)) (v_(K+1) = 0). returns T, P"""
    v = v * mask
    T = torch.cumprod(v, -1)
    nxt = torch.cat([v[..., 1:], torch.zeros_like(v[..., :1])], -1)
    return T, T * (1.0 - nxt)


def survival_tolerant(v, mask, skip=0.5):
    """one allowed missing sample at a penalty: T_k = max(T_(k-1) v_k, skip T_(k-2) v_k)"""
    v = v * mask
    K = v.shape[-1]
    Ts = [v[..., 0]]
    for k in range(1, K):
        a = Ts[k - 1] * v[..., k]
        prev = Ts[k - 2] if k >= 2 else torch.ones_like(v[..., 0])
        Ts.append(torch.maximum(a, skip * prev * v[..., k]))
    return torch.stack(Ts, -1)


def seg_dist(p, a, b):
    """distance of points p to the segments a-b (broadcasting, last dim 2)"""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0, 1)
    return (p - (a + t[..., None] * ab)).norm(dim=-1)


def candidates(v, pat, tolerant=False, cfg=CFG):
    """per ray: most probable last verified sample k* (SELECTION: index), validity nu = T_k* x prior.  v [B,N,R,K] -> kstar [B,N,R], nu [B,N,R], T [B,N,R,K]"""
    if tolerant:
        T = survival_tolerant(v, pat.mask, cfg["skip_penalty"])
        alive = (T >= cfg["alive"]) & (pat.mask > 0)
        idx = torch.arange(v.shape[-1], device=v.device)
        kstar = (alive * (idx + 1)).max(-1).values.clamp(min=1) - 1  # SELECTION
    else:
        T, P = survival(v, pat.mask)
        kstar = P.argmax(-1)  # SELECTION: most probable last verified sample (0-based)
    nu = T.gather(-1, kstar[..., None])[..., 0] * pat.prior
    return kstar, nu, T


def covered(end, committed, cmask, cfg=CFG):
    """endpoints [B,N,R,2] within cov_r of a committed point [B,N,Lc,2] (mask [B,N,Lc]) -> bool [B,N,R]"""
    d = (end[:, :, :, None, :] - committed[:, :, None, :, :]).norm(dim=-1)
    d = d.masked_fill(cmask[:, :, None, :] < 0.5, 1e9)
    return (d < cfg["cov_r"]).any(-1)


def select_children(end, nu, tip, b, cfg=CFG, relax=None, relax_seg=0.35, relax_r=0.6):
    """SELECTION: dominance redundancy (a longer comparably valid candidate whose segment passes the endpoint), then greedy endpoint NMS; returns idx [B,N,b] and ok [B,N,b].
    end [B,N,R,2], nu [B,N,R], tip [B,N,2]. Candidates are ordered by nu (ties: lower ray index), as in the numpy reference."""
    B, N, R, _ = end.shape
    reach = (end - tip[:, :, None, :]).norm(dim=-1)  # [B,N,R]
    sd = seg_dist(end[:, :, :, None, :], tip[:, :, None, None, :], end[:, :, None, :, :])  # [B,N,i,j]: end_i against the segment tip -> end_j
    eye = torch.eye(R, dtype=torch.bool, device=end.device)
    if relax is None:
        seg_thr, r_thr = cfg["nms_seg"], cfg["nms_r"]
    else:  # tips in front of a predicted fork keep near-coincident children (shallow forks): smaller dominance and endpoint-suppression radii (a selection input, no gradient)
        seg_thr = torch.where(relax, torch.full_like(reach[..., 0], relax_seg), torch.full_like(reach[..., 0], cfg["nms_seg"]))[:, :, None, None]
        r_thr = torch.where(relax, torch.full_like(reach[..., 0], relax_r), torch.full_like(reach[..., 0], cfg["nms_r"]))[:, :, None]
    dom = (reach[:, :, None, :] > reach[:, :, :, None] + 1e-6) & (nu[:, :, None, :] >= 0.5 * nu[:, :, :, None]) & (sd < seg_thr) & ~eye
    red = dom.any(-1)  # [B,N,R]
    ridx = torch.arange(R, device=end.device, dtype=nu.dtype)
    score = (nu - 1e-7 * ridx).masked_fill(red, -1e9)
    pair = (end[:, :, :, None, :] - end[:, :, None, :, :]).norm(dim=-1)  # [B,N,R,R]
    idxs, oks = [], []
    for _ in range(b):
        best = score.argmax(-1)  # SELECTION
        ok = score.gather(-1, best[..., None])[..., 0] > -1e8
        idxs.append(best)
        oks.append(ok)
        sup = pair.gather(2, best[:, :, None, None].expand(B, N, 1, R))[:, :, 0] <= r_thr
        score = score.masked_fill(sup | torch.eye(R, dtype=torch.bool, device=end.device)[best], -1e9)
    return torch.stack(idxs, -1), torch.stack(oks, -1)


def path_validity(pi_parent, nu_child):
    """bottleneck path validity pi = min over the path"""
    return torch.minimum(pi_parent, nu_child)


def beam_topm(pi, alive, M):
    """SELECTION: indices of the M alive nodes with the highest pi per frame. pi, alive [B,Nc] -> idx [B,M], ok [B,M]"""
    s = pi.masked_fill(~alive, -1.0)
    M = min(M, s.shape[1])
    val, idx = s.topk(M, dim=1)
    return idx, val > -0.5
