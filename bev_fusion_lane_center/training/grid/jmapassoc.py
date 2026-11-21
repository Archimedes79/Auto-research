"""E391 step 2 (card experiments/E391_sd-map-fusion-association): the FUSION heads that associate the champion's perceived tokens to the segments of the E390 road-graph encoding and read logical data through it.

    head = MapAssocHead(arm)        # arm: z (no map), P (plain cross-attention), F1 (cross-attention + relative-geometry bias), F2 ([token ; read] + MLP), F3 (F1 + gated residual)
    out = head(tok_batch, h, seg_valid, pa)   # tok_batch: the token keys (tok, pos, dirv, e, flags, valid); h (B, N, 64) segment embeddings of the FROZEN E390 encoder; seg_valid (B, N); pa (B, K, N, 8)
    out["assoc"] (B, K, N + 1) logits (the last slot = 'unassigned'), out["lanes"] (B, K, 4), out["route"] (B, K)

Perception untouched (D135): every token input is DETACHED before the head reads it, so no map loss can send a gradient into the token features (on the cached tokens nothing upstream is trained; the test
`test_mapassoc.py` checks the token tensors get no gradient, with a no-detach mutant). The token encoder is E389's (LN(Linear(tok)) + an MLP of pos / 50, clip(pos / 5), dirv, sigmoid(e), sigmoid(flags)).
Association: one cross-attention: logits q(x) . k(h) / sqrt(d) (+ F1 / F2 / F3: a bias MLP of the token-segment relative geometry pa) and a learned 'unassigned' key; the read r = softmax over the present
segments and 'unassigned' (its value 0) x v(h). Attributes: P / F1: MLP(r); F2: MLP([x ; r]); F3: MLP(x + sigmoid(w x + b) * W r) with b = -4 (the no-map behaviour at the start); z: MLP(x) (no map at all).
"""
import math

import torch
import torch.nn as nn

ARMS = ("z", "P", "F1", "F2", "F3")
PA_F = 8


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MapAssocHead(nn.Module):
    def __init__(self, arm, d=64, d_map=64, n_lane=4):
        super().__init__()
        assert arm in ARMS, arm
        self.arm, self.d = arm, d
        self.tok_in = nn.Sequential(nn.Linear(64, d), nn.LayerNorm(d))
        self.geo_in = mlp(2 + 2 + 2 + 1 + 8, d, d)
        if arm != "z":
            self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d_map, d), nn.Linear(d_map, d)
            self.none_k = nn.Parameter(torch.zeros(d))
            self.none_b = nn.Parameter(torch.zeros(1))
            if arm != "P":
                self.bias = mlp(PA_F, 32, 1)
            if arm == "F3":
                self.gate = nn.Linear(d, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, -4.0)
                self.res = nn.Linear(d, d)
        attr_in = 2 * d if arm == "F2" else d
        self.lanes, self.route = mlp(attr_in, d, n_lane), mlp(attr_in, d, 1)

    def token_x(self, t):
        tok = t["tok"].detach().float()
        pos, dirv = t["pos"].detach().float(), t["dirv"].detach().float()
        e, flags = t["e"].detach().float(), t["flags"].detach().float()
        g = torch.cat([pos / 50.0, (pos / 5.0).clamp(-1, 1), dirv, e.sigmoid()[..., None], flags.sigmoid()], -1)
        return self.tok_in(tok) + self.geo_in(g)

    def forward(self, t, h=None, seg_valid=None, pa=None):
        x = self.token_x(t)
        out = {}
        if self.arm == "z":
            a = x
        else:
            B, K, _ = x.shape
            q, k, v = self.q(x), self.k(h), self.v(h)
            z = q @ k.transpose(1, 2) / math.sqrt(self.d)
            if self.arm != "P":
                z = z + self.bias(pa.float())[..., 0]
            z = z.masked_fill(~seg_valid[:, None, :], float("-inf"))
            zn = (q @ self.none_k)[..., None] / math.sqrt(self.d) + self.none_b
            logits = torch.cat([z, zn], -1)
            out["assoc"] = logits
            w = torch.softmax(logits, -1)[..., :-1]
            r = w @ v
            if self.arm in ("P", "F1"):
                a = r
            elif self.arm == "F2":
                a = torch.cat([x, r], -1)
            else:
                a = x + torch.sigmoid(self.gate(x)) * self.res(r)
        out["lanes"], out["route"] = self.lanes(a), self.route(a)[..., 0]
        return out


def selftest():
    torch.manual_seed(0)
    B, K, N = 2, 6, 4
    t = dict(tok=torch.randn(B, K, 64, requires_grad=True), pos=torch.randn(B, K, 2) * 10, dirv=torch.randn(B, K, 2), e=torch.randn(B, K), flags=torch.randn(B, K, 8), valid=torch.ones(B, K, dtype=torch.bool))
    h = torch.randn(B, N, 64)
    sv = torch.tensor([[True, True, True, False], [True, True, False, False]])
    pa = torch.randn(B, K, N, PA_F)
    for arm in ARMS:
        m = MapAssocHead(arm)
        o = m(t, h, sv, pa)
        assert o["lanes"].shape == (B, K, 4) and o["route"].shape == (B, K)
        if arm != "z":
            assert o["assoc"].shape == (B, K, N + 1) and torch.isinf(o["assoc"][0, :, 3]).all() and torch.isfinite(o["assoc"][..., -1]).all()
        loss = o["lanes"].sum() + o["route"].sum() + (o["assoc"][..., :2].sum() if arm != "z" else 0)
        loss.backward()
        assert t["tok"].grad is None, f"{arm}: a gradient reached the token features (perception)"
    f3 = MapAssocHead("F3")
    assert abs(float(torch.sigmoid(f3.gate.bias)) - 0.018) < 0.001
    print("jmapassoc selftest ok")


if __name__ == "__main__":
    selftest()
