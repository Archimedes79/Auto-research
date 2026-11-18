"""B, the lane head (E273 stages B - D): nodes on a 2 m grid, links between them (cubic curves), sections (lane pieces).

    no = NodeHead()(P)                # per 2 m cell: existence e, 8 type flags, position, heading, section start s, long-link start / end scores, token
    lanes = LinkHead(cfg)(no, P)      # active nodes, candidate links with logit and handle scales, the section start of every active node
Every step is soft; the candidate sets (the top-K nodes, the long-link endpoints) are index choices without gradient through the choice.
Invariants the read-out relies on: (1) the active nodes are sorted by existence, so the valid ones form a prefix and a node id = its slot index;
(2) the candidate links are unique (src, dst) pairs with src != dst, ordered by frame.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from lg_config import Config
from lg_layers import LocalAttn, SetBlock, bezier, cell_centres, fourier_pts, sample, unit

N = 50                                         # 2 m grid


def grid_xy(device) -> torch.Tensor:
    """(1, 2, N, N) cell centres in metres (x along rows, y along columns)"""
    c = cell_centres(N, device)
    return torch.stack([c.view(N, 1).expand(N, N), c.view(1, N).expand(N, N)], 0)[None]


class NodeHead(nn.Module):
    """one node slot per 2 m cell: a token from the pyramid's maps, two layers of 7 x 7 neighbourhood attention, a first read-out, then a refinement that
    samples the 1 m / 0.5 m features at the pulled node position and re-reads offset, heading and flags"""

    def __init__(self, c2: int = 144, c1: int = 48, c05: int = 32, d: int = 64):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(c2 + c1 + c05 + 2 + 16 + 8, d, 1), nn.GroupNorm(8, d), nn.GELU())
        self.att = nn.ModuleList([LocalAttn(d, N), LocalAttn(d, N)])
        self.head1 = nn.Conv2d(d, 16, 1)
        self.fine = nn.Sequential(nn.Conv2d(d + c1 + c05 + 4, d, 1), nn.GELU(), nn.Conv2d(d, d, 1))
        self.head2 = nn.Conv2d(d, 16, 1)

    def forward(self, P: dict) -> dict:
        B = P["feat2"].shape[0]
        c = grid_xy(P["feat2"].device).expand(B, -1, -1, -1)
        cen, ev = P["centre1"][:, :1].sigmoid(), P["centre1"][:, 6:12].sigmoid()
        pooled = torch.cat([F.avg_pool2d(cen, 2), F.max_pool2d(cen, 2), F.avg_pool2d(ev, 2)], 1)
        code = fourier_pts(c[:1].permute(0, 2, 3, 1)).permute(0, 3, 1, 2).expand(B, -1, -1, -1)
        t = self.inp(torch.cat([P["feat2"], F.avg_pool2d(P["feat1"], 2), F.avg_pool2d(P["feat05"], 4), P["road2"].sigmoid(), code, pooled], 1))
        for a in self.att:
            t = a(t)
        o1 = self.head1(t)
        off1 = torch.tanh(o1[:, 9:11])
        p1 = (c + off1).permute(0, 2, 3, 1)
        fine = [sample(P["feat1"], p1, border=True).permute(0, 3, 1, 2), sample(P["feat05"], p1, border=True).permute(0, 3, 1, 2)]
        t2 = t + self.fine(torch.cat([t, *fine, off1, unit(o1[:, 11:13])], 1))
        o = o1 + self.head2(t2)
        return dict(e=o[:, 0], flags=o[:, 1:9], dir_raw=o[:, 11:13], s=o[:, 13], uout=o[:, 14], uin=o[:, 15], tok=t2, pos=c + torch.tanh(o[:, 9:11]))


def gather(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """m (B, C, N, N), idx (B, K) flat cells -> (B, K, C)"""
    B, C = m.shape[:2]
    return m.flatten(2).gather(2, idx[:, None, :].expand(B, C, -1)).transpose(1, 2)


class LinkHead(nn.Module):
    """links between the active nodes: short links (nearby, not behind the node) and long links (through junctions). Logit = MLP(tokens after two
    set-attention layers, geometry, centre-lane evidence along the link's cubic curve); handle scales 1/3 for short links, learned for long ones."""

    def __init__(self, cfg: Config, d: int = 64, hidden: int = 128):
        super().__init__()
        self.cfg = cfg
        self.nin = nn.Sequential(nn.Linear(16 + 2 + 11, d), nn.GELU(), nn.Linear(d, d))
        self.ctx = nn.ModuleList([SetBlock(d), SetBlock(d)])
        self.scale = nn.Sequential(nn.Linear(2 * d + 9, 64), nn.GELU(), nn.Linear(64, 2))
        self.mlp = nn.Sequential(nn.Linear(2 * d + 14, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, no: dict, P: dict) -> dict:
        """-> act (B, K) cells, valid, e (existence probability), pos, dir (B, K, 2), tok (B, K, d); frame, src, dst, long, logit, sa, sb (one entry per
        candidate link); start (B, K) the section start of every node"""
        K = min(self.cfg.max_nodes, N * N)
        e, act = no["e"].sigmoid().flatten(1).topk(K, dim=1)
        lanes = dict(act=act, valid=e > self.cfg.node_min, e=e, pos=gather(no["pos"], act), dir=unit(gather(no["dir_raw"], act), dim=-1))
        scores = torch.cat([no["e"][:, None], no["flags"], no["uout"][:, None], no["uin"][:, None]], 1).sigmoid()
        tok = gather(no["tok"], act) + self.nin(torch.cat([fourier_pts(lanes["pos"]), lanes["dir"], gather(scores, act)], -1))
        for blk in self.ctx:
            tok = blk(tok, ~lanes["valid"])
        lanes["frame"], lanes["src"], lanes["dst"] = self.candidates(no, lanes)
        lanes.update(self.score(lanes, tok, P["centre1"][:, :1].sigmoid()))
        lanes["start"] = section_start(lanes, no["s"])
        return lanes

    def candidates(self, no: dict, lanes: dict) -> tuple:
        """(frame, src, dst) of the candidate links: short links within short_cells cells and not behind the source; long links from the long_ends
        strongest starts to the long_ends strongest ends, short_reach .. long_max apart, inside the forward cone"""
        cfg, act, valid, pos, dirv = self.cfg, lanes["act"], lanes["valid"], lanes["pos"], lanes["dir"]
        K = act.shape[1]
        ci, cj = act // N, act % N
        both = valid[:, :, None] & valid[:, None, :] & ~torch.eye(K, dtype=torch.bool, device=act.device)[None]
        dp = pos[:, None, :, :] - pos[:, :, None, :]                                         # [b, a, c] = pos_c - pos_a
        along, dist = (dp * dirv[:, :, None, :]).sum(-1), dp.norm(dim=-1)
        near = ((ci[:, :, None] - ci[:, None, :]).abs() <= cfg.short_cells) & ((cj[:, :, None] - cj[:, None, :]).abs() <= cfg.short_cells)
        ends = []
        for u in (no["uout"], no["uin"]):
            top = u.sigmoid().flatten(1).gather(1, act).topk(min(cfg.long_ends, K), dim=1)[1]
            ends.append(torch.zeros_like(valid).scatter_(1, top, True) & valid)
        long = both & ends[0][:, :, None] & ends[1][:, None, :] & (dist > cfg.short_reach) & (dist <= cfg.long_max) & (along / dist.clamp_min(1e-6) > cfg.cos_cone)
        return ((both & near & (along > cfg.short_behind)) | long).nonzero(as_tuple=True)

    def score(self, lanes: dict, tok: torch.Tensor, cen: torch.Tensor) -> dict:
        """link logit and handle scales of every candidate; the evidence = the centre-lane probability `cen` along the link's curve"""
        pb, pa, pc, pos, dirv = lanes["frame"], lanes["src"], lanes["dst"], lanes["pos"], lanes["dir"]
        p0, p1, h0, h1 = pos[pb, pa], pos[pb, pc], dirv[pb, pa], dirv[pb, pc]
        d_ = p1 - p0
        dd, al, ac = d_.norm(dim=-1), (d_ * h0).sum(-1), d_[:, 0] * h0[:, 1] - d_[:, 1] * h0[:, 0]
        long = dd > self.cfg.short_reach
        geo = torch.stack([al / 8, ac / 8, dd / 8, al / 50, ac / 50, dd / 50, (h0 * h1).sum(-1), h0[:, 0] * h1[:, 1] - h0[:, 1] * h1[:, 0], long.float()], -1)
        x0 = torch.cat([tok[pb, pa], tok[pb, pc], geo], -1)
        sc = 0.9 * torch.sigmoid(self.scale(x0))
        third = torch.full_like(sc[:, 0], 1 / 3)
        sa, sb = torch.where(long, sc[:, 0], third), torch.where(long, sc[:, 1], third)
        curve = bezier(p0, h0, p1, h1, sa[:, None], sb[:, None], self.cfg.link_samples)
        ev = torch.zeros(len(pb), self.cfg.link_samples, device=pos.device)
        for b in range(pos.shape[0]):                                                         # the candidates are ordered by frame
            sel = (pb == b).nonzero(as_tuple=True)[0]
            ev[sel] = sample(cen[b:b + 1], curve[sel][None], border=True)[0, ..., 0]
        evf = torch.stack([ev.mean(1), ev.min(1)[0], torch.sigmoid((0.2 - ev) * 20).mean(1)], -1)
        return dict(long=long, logit=self.mlp(torch.cat([x0, sa[:, None], sb[:, None], evf], -1))[:, 0], sa=sa, sb=sb)


def section_start(lanes: dict, s_logit: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
    """(B, K) the section of every active node = the node its section starts at: a soft absorbing walk backwards over the links (absorbed at a node
    with probability s, else a step to a predecessor drawn from the softmax over its incoming link logits and 'none'), M = (I - Q)^-1 R.
    The argmax over M is a NAMED HARD STEP of the output (the membership M itself is soft)."""
    act, valid = lanes["act"], lanes["valid"]
    B, K = valid.shape
    s = s_logit.flatten(1).gather(1, act).sigmoid().clamp(ridge, 1 - 1e-4)
    s = torch.where(valid, s, torch.ones_like(s))
    L = torch.full((B, K, K), float("-inf"), device=act.device).index_put((lanes["frame"], lanes["dst"], lanes["src"]), lanes["logit"])
    Pb = torch.cat([L, torch.zeros(B, K, 1, device=act.device)], -1).softmax(-1)[..., :K] * valid[:, :, None] * valid[:, None, :]
    M = torch.linalg.solve(torch.eye(K, device=act.device)[None] - (1 - s)[:, :, None] * Pb, torch.diag_embed(s)).clamp_min(0)
    return (M / M.sum(-1, keepdim=True).clamp_min(1e-9)).argmax(-1)
