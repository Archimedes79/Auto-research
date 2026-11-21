"""E390 step 1 (card experiments/E390_sd-map-logical-road-graph): the ENCODER of the SD road graph and the three probe heads; no perception input.

    enc = SDGraphEncoder(d=64, heads=4, layers=2)     # layers 0 = the no-relation ablation (arm n): each node sees only its own features
    h = enc(x, rel, pair, valid)                       # x (B, N, NODE_F), rel (B, R, N, N) bool, pair (B, N, N, EDGE_F), valid (B, N) bool -> (B, N, d)
    probe = ProbeHeads(d)                              # p1(h) (B, N, 4) lane-class logits; p2(h, pair) (B, N, N) link logits; p3(h) (B, N) on-route logits
    batch = collate([featurized frames])               # padding to the largest frame of the batch

The attention of a layer: softmax over the valid nodes of q_i . k_j / sqrt(dh) + b_ij, with b_ij = an MLP of [the relation one-hot of (i, j) ; the pair features of (i, j)] (one bias per head); a node always attends to
itself. Everything the encoder reads is invariant under a rigid motion of the frame (the inputs are: node features in the node's own frame, relations, pair features in a node's local frame): the embedding is
invariant by construction (tested); the ego-frame pose is NOT read here (step 2).
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Layer(nn.Module):
    def __init__(self, d, heads, rel, edge_f):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = mlp(rel + edge_f, 32, heads)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, 2 * d, d)

    def forward(self, x, rel, pair, valid):
        B, N, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        b = self.bias(torch.cat([rel.permute(0, 2, 3, 1).float(), pair], -1)).permute(0, 3, 1, 2)          # (B, heads, N, N)
        z = q @ k.transpose(-1, -2) / math.sqrt(self.dh) + b
        mask = valid[:, None, None, :] | torch.eye(N, dtype=torch.bool, device=x.device)[None, None]
        z = z.masked_fill(~mask, float("-inf"))
        a = torch.softmax(z, -1)
        y = (a @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.out(y)
        return x + self.ff(self.n2(x))


class SDGraphEncoder(nn.Module):
    def __init__(self, node_f, edge_f, rel=5, d=64, heads=4, layers=2):
        super().__init__()
        self.inp = mlp(node_f, d, d)
        self.layers = nn.ModuleList([Layer(d, heads, rel, edge_f) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, rel, pair, valid):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h, rel, pair, valid)
        return self.norm(h) * valid[..., None]


class ProbeHeads(nn.Module):
    def __init__(self, d, edge_f, n_lane=4, pa_f=8):
        super().__init__()
        self.p1 = mlp(d, d, n_lane)
        self.p2 = mlp(2 * d + edge_f, d, 1)
        self.p3 = mlp(d, d, 1)
        self.p6 = mlp(d + pa_f, d, 1)                                       # D135: the association score of (proxy, segment) from the segment's embedding and their relative geometry
        self.p6_none = mlp(3, 16, 1)                                        # the 'unassigned' score of a proxy from its nearest present segment (distance, cos, lateral: signed)

    def lanes(self, h):
        return self.p1(h)

    def links(self, h, pair):
        B, N, d = h.shape
        z = torch.cat([h[:, :, None].expand(B, N, N, d), h[:, None].expand(B, N, N, d), pair], -1)
        return self.p2(z)[..., 0]

    def route(self, h):
        return self.p3(h)[..., 0]

    def assoc(self, h, pa, valid):
        """D135 (P6): (B, M, N + 1) logits of the assignment of each proxy to the present segments and to 'unassigned' (the last slot); h (B, N, d), pa (B, M, N, PA_F), valid (B, N) the present segments"""
        B, M, N, _ = pa.shape
        z = self.p6(torch.cat([h[:, None].expand(B, M, N, h.shape[-1]), pa], -1))[..., 0]
        z = z.masked_fill(~valid[:, None, :], float("-inf"))
        dist = pa[..., 0].masked_fill(~valid[:, None, :], 1e3)
        j = dist.argmin(-1, keepdim=True)
        near = torch.cat([dist.gather(-1, j), pa[..., 1].gather(-1, j), pa[..., 4].gather(-1, j)], -1)        # signed (the critic of v3: the opposite direction shares the divider line)
        return torch.cat([z, self.p6_none(near)], -1)


PAIR_KEYS = ("pair", "link", "dec")                                           # (N, N, ...) per frame; "rel" is (R, N, N); "pa" is (M, N, ...); PROX_KEYS are per proxy (M, ...); every other key is per node (N, ...)
PROX_KEYS = ("assoc",)


def collate(items):
    """items = dicts of `lane_sdgraph_data.featurize` (+ the targets of `sdgraph_probe.targets`, numpy) -> padded torch tensors and valid (B, N); the padding is decided by the key NAME (code critic: a per-node key
    with N columns looked like a pair matrix); integer targets pad with -1, booleans with False, floats with 0"""
    N = max(len(it["x"]) for it in items)
    M = max([len(it["pa"]) for it in items if "pa" in it] or [0])
    B = len(items)
    out = {"valid": torch.zeros(B, N, dtype=torch.bool)}
    for k in items[0]:
        a0 = np.asarray(items[0][k])
        fill = False if a0.dtype == bool else (-1 if a0.dtype.kind in "iu" else 0)
        if k == "rel":
            arr = np.zeros((B, a0.shape[0], N, N), bool)
            for b, it in enumerate(items):
                n = len(it["x"])
                arr[b, :, :n, :n] = it[k]
        elif k == "pa":
            arr = np.zeros((B, M, N, a0.shape[-1]), a0.dtype)
            for b, it in enumerate(items):
                arr[b, : it[k].shape[0], : it[k].shape[1]] = it[k]
        elif k in PROX_KEYS:
            arr = np.full((B, M), -2, np.int64)                                # -2 = no proxy (ignored), -1 = unassigned
            for b, it in enumerate(items):
                arr[b, : len(it[k])] = it[k]
        elif k in PAIR_KEYS:
            arr = np.full((B, N, N) + a0.shape[2:], fill, a0.dtype)
            for b, it in enumerate(items):
                n = len(it["x"])
                arr[b, :n, :n] = it[k]
        else:
            arr = np.full((B, N) + a0.shape[1:], fill, a0.dtype)
            for b, it in enumerate(items):
                arr[b, : len(it["x"])] = it[k]
        out[k] = torch.as_tensor(arr)
    for b, it in enumerate(items):
        out["valid"][b, : len(it["x"])] = True
    if "present" in out:
        out["valid"] &= out["present"]                                          # an absent map segment is masked like padding
    return out


def selftest():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    import lane_sdgraph_data as SG
    torch.manual_seed(0)
    G = SG.toy()
    a, _ = SG.perturb(G, np.random.default_rng(0))
    b, _ = SG.perturb(G, np.random.default_rng(0), rigid=(12.0, -5.0, 0.4))
    enc = SDGraphEncoder(SG.NODE_F, SG.EDGE_F).eval()
    ba, bb = collate([SG.featurize(a)]), collate([SG.featurize(b)])
    with torch.no_grad():
        ha = enc(ba["x"], ba["rel"], ba["pair"].float(), ba["valid"])
        hb = enc(bb["x"], bb["rel"], bb["pair"].float(), bb["valid"])
    assert ha.shape == (1, 5, 64) and (ha - hb).abs().max() <= 1e-4 * ha.abs().mean(), "the embedding is not rigid-motion invariant"
    # padding: a frame padded next to a larger one gives the same embedding
    big = SG.toy()
    big = dict(big, kind=np.r_[big["kind"], 0], geo=big["geo"] + [[np.stack([np.linspace(30, 40, 11), np.zeros(11)], 1)]], lanes=np.r_[big["lanes"], 0], route=np.r_[big["route"], -1])
    c, _ = SG.perturb(big, np.random.default_rng(0))
    bc = collate([SG.featurize(a), SG.featurize(c)])
    with torch.no_grad():
        hc = enc(bc["x"], bc["rel"], bc["pair"].float(), bc["valid"])
    assert (hc[0, :5] - ha[0]).abs().max() < 1e-5 and (hc[0, 5] == 0).all(), "padding changes the embedding"
    bd = collate([SG.featurize(c), SG.featurize(a)])                         # a 6-node frame FIRST: its pose (6, 6) must not be taken for a pair matrix
    assert bd["pose"].shape == (2, 6, 6) and bd["pair"].shape == (2, 6, 6, SG.EDGE_F) and (bd["pose"][1, 5] == 0).all()
    print("jsdgraph selftest ok")


if __name__ == "__main__":
    selftest()
