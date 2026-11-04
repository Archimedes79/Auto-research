"""Stage E of the lane head (E273): the map encoder and the assignment of the nodes to the map pieces.

    MapEncoder(d)(map_pos, map_tan, map_frac, map_pidx, map_pfeat, map_adj, kmask, tmask) -> dict(piece (B, K, d), tok (B, T, d))
    Assigner(d)(node_tok, node_pos, node_dir, node_w, node_valid, map, enc, rounds=2) -> dict(P_tok (B, A, T + 1), P_piece (B, A, K + 1), P_road (B, A, R + 1), shift (B, 2), yaw (B,))

The map = the lane GT pieces of the frame with simulated errors (`map_sim.make_map`): tokens every 2 m (position, tangent, arclength fraction), piece features (kind, lane count, lane index,
numbers of successors / predecessors, length, open ends) and the five relations succ / pred / left / right / opposite. The encoder is a 3-layer graph transformer over the PIECES whose attention
runs along those relations (relation bias per head), the token embedding = a token MLP + its piece embedding.
The assignment is a cross-attention from the nodes (rows: the perception structure is kept) to the map tokens plus a dustbin:
    logit(i, t) = - [ |p_i - q_t|^2 / sigma^2 + lam (1 - h_i . tau_t) ] + (W_q tok_i) . (W_k emb_t) / sqrt(d),   dustbin: - (c_bin + MLP(tok_i))
the first part is E272's geometric assigner (sigma 1.5, lam 2, c_bin 4: the start equals the control g, the learned part starts near zero); P(piece) = sum of P(token) over the tokens of the piece.
`rounds` rounds of soft registration: a weighted Procrustes (rotation + translation of the map tokens, weights = node weight x P(token)) moves the map towards the nodes and the logits are
recomputed (the shift is clamped to +-5 m, the rotation to +-6 degrees). Road level: P(road) = sum of P(piece) over the pieces of the road.
Frame: lane frame (x right, y forward) for nodes and map alike; nothing is converted.
Run the self-test: python qassign.py
"""
import math

import torch
import torch.nn as nn


def fourier_pts(p, nf=4):
    k = (2.0 ** torch.arange(nf, device=p.device, dtype=p.dtype)) * math.pi / 50.0
    a = p[..., None] * k                                                                       # (..., 2, nf)
    return torch.cat([a.sin(), a.cos()], -1).flatten(-2)                                       # (..., 4 nf)


class RelAttn(nn.Module):
    def __init__(self, d, heads=4, rel=5):
        super().__init__()
        self.h, self.d = heads, d
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.rel = nn.Parameter(torch.zeros(rel, heads))
        self.selfb = nn.Parameter(torch.zeros(heads))

    def forward(self, x, adj, kmask):
        B, K, d = x.shape
        h, dh = self.h, d // self.h
        q, k, v = self.qkv(self.n1(x)).view(B, K, 3, h, dh).unbind(2)
        logit = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(dh)
        logit = logit + torch.einsum("brij,rh->bhij", adj.float(), self.rel) + torch.eye(K, device=x.device)[None, None] * self.selfb.view(1, h, 1, 1)
        allowed = (adj.any(1) | torch.eye(K, dtype=torch.bool, device=x.device)[None]) & kmask[:, None, :] & kmask[:, :, None]
        logit = logit.masked_fill(~allowed[:, None], float("-inf"))
        att = torch.nan_to_num(logit.softmax(-1), nan=0.0)
        out = torch.einsum("bhij,bjhd->bihd", att, v).reshape(B, K, d)
        x = x + self.proj(out)
        return x + self.ffn(self.n2(x))


class MapEncoder(nn.Module):
    def __init__(self, d=64, layers=3, heads=4):
        super().__init__()
        self.d = d
        self.tok = nn.Sequential(nn.Linear(5 + 16, d), nn.GELU(), nn.Linear(d, d))
        self.pie = nn.Sequential(nn.Linear(d + 12, d), nn.GELU(), nn.Linear(d, d))
        self.layers = nn.ModuleList([RelAttn(d, heads) for _ in range(layers)])

    def forward(self, map_pos, map_tan, map_frac, map_pidx, map_pfeat, map_adj, kmask, tmask):
        B, T, _ = map_pos.shape
        K = map_pfeat.shape[1]
        te = self.tok(torch.cat([map_pos / 50.0, map_tan, map_frac[..., None], fourier_pts(map_pos)], -1)) * tmask[..., None]
        onehot = ((map_pidx[:, None, :] == torch.arange(K, device=map_pos.device)[None, :, None]) & tmask[:, None, :]).float()          # (B, K, T)
        pooled = onehot @ te / onehot.sum(-1, keepdim=True).clamp_min(1.0)
        x = self.pie(torch.cat([pooled, map_pfeat], -1)) * kmask[..., None]
        for l in self.layers:
            x = l(x, map_adj, kmask) * kmask[..., None]
        tok = te + x.gather(1, map_pidx.clamp_min(0)[..., None].expand(-1, -1, self.d)) * tmask[..., None]
        return dict(piece=x, tok=tok, onehot=onehot)


class Assigner(nn.Module):
    def __init__(self, d=64, sigma=1.5, lam=2.0, c_bin=4.0):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.lam = nn.Parameter(torch.tensor(float(lam)))
        self.c_bin = nn.Parameter(torch.tensor(float(c_bin)))
        self.wq, self.wk = nn.Linear(d, d), nn.Linear(d, d)
        self.dust = nn.Linear(d, 1)
        for m in (self.wq, self.wk, self.dust):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        self.d = d

    def _logits(self, q, k, pos, dirv, mpos, mtan, tmask, tok_i):
        d = pos[:, :, None, :] - mpos[:, None, :, :]
        cost = (d * d).sum(-1) / torch.exp(2 * self.log_sigma) + self.lam * (1.0 - (dirv[:, :, None, :] * mtan[:, None, :, :]).sum(-1))
        logit = -cost + torch.einsum("bad,btd->bat", q, k) / math.sqrt(self.d)
        logit = logit.masked_fill(~tmask[:, None, :], float("-inf"))
        dust = -(self.c_bin + self.dust(tok_i))
        return torch.cat([logit, dust], -1)

    def forward(self, node_tok, node_pos, node_dir, node_w, node_valid, mp, enc, rounds=2):
        B, A, _ = node_tok.shape
        tmask = mp["tmask"]
        mpos0, mtan0 = mp["map_pos"], mp["map_tan"]
        q, k = self.wq(node_tok), self.wk(enc["tok"])
        w0 = node_w * node_valid.float()
        th = torch.zeros(B, device=node_tok.device)
        u = torch.zeros(B, 2, device=node_tok.device)
        mpos, mtan = mpos0, mtan0
        T = mpos0.shape[1]
        for r in range(rounds + 1):
            P = self._logits(q, k, node_pos, node_dir, mpos, mtan, tmask, node_tok).softmax(-1)
            if r == rounds:
                break
            W = (w0[:, :, None] * P[..., :T] * tmask[:, None, :].float())                       # (B, A, T)
            Ws = W.sum((1, 2)).clamp_min(1e-3)
            pbar = (W.sum(2)[..., None] * node_pos).sum(1) / Ws[:, None]
            tbar = (W.sum(1)[..., None] * mpos0).sum(1) / Ws[:, None]
            pc, tc = node_pos - pbar[:, None], mpos0 - tbar[:, None]
            sxx = torch.einsum("bat,bax,btx->b", W, pc, tc)
            sxy = torch.einsum("bat,bay,btx->b", W, pc, tc)
            syx = torch.einsum("bat,bax,bty->b", W, pc, tc)
            th = torch.atan2(sxy - syx, sxx + torch.einsum("bat,bay,bty->b", W, pc, tc) + 1e-6).clamp(-math.radians(6), math.radians(6))
            c, s = th.cos(), th.sin()
            R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)           # (B, 2, 2)
            u = (pbar - torch.einsum("bij,bj->bi", R, tbar)).clamp(-5.0, 5.0)
            mpos = torch.einsum("bij,btj->bti", R, mpos0) + u[:, None]
            mtan = torch.einsum("bij,btj->bti", R, mtan0)
        P_tok = P
        K = mp["map_pfeat"].shape[1]
        P_piece = torch.cat([P_tok[..., :T] @ enc["onehot"].transpose(1, 2), P_tok[..., T:]], -1)                                 # (B, A, K + 1)
        R_n = int(mp["map_road"].max().item()) + 1 if mp["map_road"].numel() else 1
        road1h = torch.zeros(B, K, max(R_n, 1), device=P.device)
        ok = mp["map_road"] >= 0
        road1h.scatter_(2, mp["map_road"].clamp_min(0)[..., None], ok[..., None].float())
        P_road = torch.cat([P_piece[..., :K] @ road1h, P_piece[..., K:]], -1)
        return dict(P_tok=P_tok, P_piece=P_piece, P_road=P_road, shift=u, yaw=th)


def _selftest():
    torch.manual_seed(0)
    B, A, K, d = 2, 40, 3, 16
    # three straight pieces along y at x = -4, 0, 4; tokens every 2 m
    ys = torch.arange(-40.0, 40.1, 2.0)
    T1 = len(ys)
    T = 3 * T1
    pos, tan, frac, pidx = [], [], [], []
    for k, x0 in enumerate((-4.0, 0.0, 4.0)):
        pos.append(torch.stack([torch.full_like(ys, x0), ys], 1))
        tan.append(torch.tensor([0.0, 1.0]).expand(T1, 2))
        frac.append(torch.linspace(0, 1, T1))
        pidx.append(torch.full((T1,), k))
    mp = dict(map_pos=torch.cat(pos)[None].repeat(B, 1, 1), map_tan=torch.cat(tan)[None].repeat(B, 1, 1), map_frac=torch.cat(frac)[None].repeat(B, 1), map_pidx=torch.cat(pidx)[None].repeat(B, 1),
              map_pfeat=torch.randn(B, K, 12), map_adj=torch.zeros(B, 5, K, K, dtype=torch.bool), kmask=torch.ones(B, K, dtype=torch.bool), tmask=torch.ones(B, T, dtype=torch.bool),
              map_road=torch.tensor([[0, 0, 1]] * B))
    mp["map_adj"][:, 2, 0, 1] = True
    mp["map_adj"][:, 3, 1, 0] = True
    enc_m, asg = MapEncoder(d=d), Assigner(d=d)
    enc = enc_m(mp["map_pos"], mp["map_tan"], mp["map_frac"], mp["map_pidx"], mp["map_pfeat"], mp["map_adj"], mp["kmask"], mp["tmask"])
    assert enc["piece"].shape == (B, K, d) and enc["tok"].shape == (B, T, d)
    # nodes: 40 nodes on piece 1 (x = 0) but the map is shifted by +1.2 m in x and rotated 2 degrees: the registration should recover most of it
    ny = torch.linspace(-38, 38, A)
    npos = torch.stack([torch.zeros(A), ny], 1)[None].repeat(B, 1, 1).requires_grad_(True)
    ndir = torch.tensor([0.0, 1.0]).expand(B, A, 2).clone()
    th = math.radians(2.0)
    Rm = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    shifted = dict(mp)
    shifted["map_pos"] = mp["map_pos"] @ Rm.T + torch.tensor([1.2, 0.0])
    shifted["map_tan"] = mp["map_tan"] @ Rm.T
    ntok = torch.randn(B, A, d, requires_grad=True)
    out0 = asg(ntok, npos, ndir, torch.ones(B, A), torch.ones(B, A, dtype=torch.bool), shifted, enc, rounds=0)
    out2 = asg(ntok, npos, ndir, torch.ones(B, A), torch.ones(B, A, dtype=torch.bool), shifted, enc, rounds=2)
    a0, a2 = float(out0["P_piece"][..., 1].mean()), float(out2["P_piece"][..., 1].mean())
    print("mass on the right piece without / with registration:", round(a0, 3), round(a2, 3), "| shift", out2["shift"][0].tolist(), "yaw deg", float(out2["yaw"][0]) * 57.3)
    assert a2 >= a0 - 1e-3 and abs(float(out2["P_piece"].sum(-1).mean()) - 1) < 1e-4
    assert out2["P_road"].shape[-1] == 3 and out2["P_piece"].shape == (B, A, K + 1)
    loss = -torch.log(out2["P_piece"][..., 1].clamp_min(1e-6)).mean()
    loss.backward()
    dead = [n for n, p in list(enc_m.named_parameters()) + list(asg.named_parameters()) if p.grad is None]
    assert not dead, dead
    assert npos.grad is not None and float(npos.grad.abs().sum()) > 0 and float(ntok.grad.abs().sum()) > 0
    print("qassign selftest ok")


if __name__ == "__main__":
    _selftest()
