"""The learned attachment of the connector ends (E320): every keypoint chooses softly among the nodes of B in reach, or 'no attachment'.

    ends = Attach(cfg)(tok, kp, no)     # pos (B, 2K, 2), dir (B, 2K, 2), p (B, 2K) = probability of an attachment
Candidates: the cfg.attach_nodes nearest nodes of B's node head within attach_reach and attach_angle whose existence is above attach_node_min
(an index choice without gradient). Score = MLP(offset and heading of the node in the keypoint's frame, the node's existence and type scores, the
node token, the keypoint kind and token); a learned 'no attachment' logit competes in the softmax. The attached end is the softmax-weighted mean of
the keypoint and the candidates (position and heading): it moves to a node when one weight dominates and stays at the keypoint when nothing is in reach.
The read-out with these ends (Config.attached_ends) still wires the connector to B's pieces by the proximity rule (`lg_readout`).
"""
import math

import torch
import torch.nn as nn

from lg_config import Config


class Attach(nn.Module):
    def __init__(self, cfg: Config, d: int = 64, tok: int = 64, hid: int = 64):
        super().__init__()
        self.cfg = cfg
        self.tp, self.kpj = nn.Linear(tok, 32), nn.Linear(d, 32)
        self.mlp = nn.Sequential(nn.Linear(5 + 12 + 1 + 32 + 32, hid), nn.GELU(), nn.Linear(hid, 1))
        self.dust = nn.Linear(d, 1)

    def forward(self, t: torch.Tensor, kp: dict, nodes: dict) -> dict:
        cfg, pos, D = self.cfg, kp["pos"], kp["dir"]
        B, n, M, R = t.shape[0], t.shape[1], cfg.attach_nodes, cfg.attach_reach
        flat = lambda x: x.flatten(2).transpose(1, 2)  # noqa: E731     (B, C, 50, 50) -> (B, 2500, C)
        npos, ndir = flat(nodes["pos"]), flat(nodes["dir_raw"])
        ndir = ndir / ndir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pe = nodes["e"].sigmoid().flatten(1)
        nf = torch.cat([pe[..., None], flat(nodes["flags"].sigmoid()), flat(torch.stack([nodes["uout"], nodes["uin"], nodes["s"]], 1).sigmoid())], -1)
        with torch.no_grad():                                                                       # the candidate set
            dist = (npos[:, None] - pos[:, :, None]).norm(dim=-1)                                   # (B, 2K, 2500)
            gate = (dist <= R) & ((ndir[:, None] * D[:, :, None]).sum(-1) >= math.cos(math.radians(cfg.attach_angle))) & (pe[:, None] > cfg.attach_node_min)
            val, idx = torch.where(gate, dist, torch.full_like(dist, float("inf"))).topk(M, dim=2, largest=False)
            cval = torch.isfinite(val)
        pick = lambda x: x.gather(1, idx.reshape(B, -1, 1).expand(-1, -1, x.shape[-1])).reshape(B, n, M, x.shape[-1])  # noqa: E731
        cpos, cdir, cf, ct = pick(npos), pick(ndir), pick(nf), pick(flat(nodes["tok"]))
        rel = cpos - pos[:, :, None]
        nrm = torch.stack([-D[..., 1], D[..., 0]], -1)[:, :, None]
        Dn = D[:, :, None]
        geo = torch.stack([(rel ** 2).sum(-1).add(1e-8).sqrt() / R, (rel * Dn).sum(-1) / R, (rel * nrm).sum(-1) / R,
                           (cdir * Dn).sum(-1), cdir[..., 0] * Dn[..., 1] - cdir[..., 1] * Dn[..., 0]], -1)
        feat = torch.cat([geo, cf, kp["kind"][:, :, None, None].expand(B, n, M, 1), self.tp(ct), self.kpj(t)[:, :, None].expand(B, n, M, 32)], -1)
        logits = torch.cat([self.dust(t), self.mlp(feat)[..., 0].masked_fill(~cval, float("-inf"))], -1)    # index 0 = no attachment
        w = logits.softmax(-1)
        w0, wc = w[..., :1], w[..., 1:, None]
        adir = w0 * D + (wc * cdir).sum(2)
        return dict(pos=w0 * pos + (wc * cpos).sum(2), dir=adir / adir.norm(dim=-1, keepdim=True).clamp_min(1e-6), p=1.0 - w0[..., 0])
