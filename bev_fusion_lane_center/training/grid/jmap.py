"""E374 (stage S3 of the token champion, D113): ONE cross-attention block from the tokens of a frame to the SD-MAP tokens of that frame (road-level samples every 2 m of the carriageway centre lines: position, the direction of travel, the lane count; `data/lane_map_data.py`).
The head of E365 (`jtoken.JTokenHead`) calls it after self-attention layer 2 (S3 hook: `map_block`, `forward(batch, past=None, maps=None)`).

    block = MapBlock(d=64, heads=4)                          # 20 264 parameters; the output projection starts at ZERO: the block is the identity at the start
    head.map_block = block                                   # the hook: x = block(x, pos, dirv, valid, maps) after layer 2 when forward(batch, maps=maps) gets maps ({} = no map)
    maps = dict(pos (B, M, 2), dir (B, M, 2), count (B, M), valid (B, M))      # the batch['maps'] of `lane_map_data`: pos in the lane frame, dir the unit direction of travel, count the lane count (0 = unknown), valid False = padding
    g = map_geometry(pos, dirv, maps["pos"], maps["dir"], maps["count"])       # (B, K, M, 7) relative geometry of every (token, map sample) pair
    meter = UseMeter(block); ...; meter.take()               # `jpast.UseMeter` on this block: dict(u = the kill measure v, u_none = v_none, n, n_other); the trainer logs them as v / v_none

MapBlock.forward(x (B, K, D), pos, dirv (B, K, 2) the tokens, valid (B, K) bool, maps) -> x (B, K, D). Pre-norm: LayerNorm of x for the queries; the map token embedding MLP([pos / 50, Fourier(pos), dir, count / 4, count known]) -> D, then LayerNorm for keys and values (one embedding of the
sample's own fields: nothing of the tokens enters it). Per head: logit = q.k / sqrt(dh) + bias_h(map_geometry) for the map samples (padding samples masked at NEG in float32) and ONE learned none key (k_none, v_none and an own bias) per head; softmax over {the map samples} + {none}; the heads are
concatenated and projected (zero-initialised); residual add; padding queries (valid False) keep x. `maps` = {} / None / all samples padding: the softmax has the none key only (the output is the none value through the projection: the identity only while the projection is zero). The 7 features of
`map_geometry`, in the frame of the TOKEN (its heading t, its left normal n): cos and sin of the angle from the token heading to the direction of travel of the sample (sin = cross(t, u): the sign of `jpast.cross_geometry`), the offset (sample - token) along t and along n / 20, the distance / 20, the lane count / 4
(0 = unknown) and a flag 'the lane count is known' (count > 0). The block reads only pos, dir, count, valid of `maps` and pos, dirv, valid of the head: no GT field, no id of the lane GT.
`MapBlock.use` is the kill measure of the card (`jpast.PastBlock.use`, the same code: it only needs `delta` and a dict of tensors with the batch on axis 0): v = RMS(block(x, map) - block(x, the map of ANOTHER SAMPLE of the batch)) / RMS(x) on the valid tokens, the other sample = every tensor of `maps` rolled by one along the batch axis;
it is 0 when the block ignores the CONTENT of the map. v_none = against the none key only is reported next to it and is NOT a test of use (any content-free attention satisfies it). Tests: grid/test_jmap.py (hand-computed values, a loop reference of the block, the invariances, a block that ignores the map content, the
real head with the hook, the mutants)
"""
import math
import os
import sys

import torch
import torch.nn as nn

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training/grid", "training/data", "training/data/prep"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jpast as JP  # noqa: E402
import jtoken as JK  # noqa: E402
from jpast import SUMS, UseMeter  # noqa: E402,F401  (the kill measure's hook and sums: the same code as for the past)

D = 64
NGEO = 7
NEMB = 2 + 4 * JK.BANDS + 2 + 1 + 1                                          # pos / HALF, its Fourier features, dir, count / 4, count known
NEG = -1e4                                                                   # the fill of a masked logit (as `jtoken.NEG`)
HALF = JK.JT.HALF


def map_features(maps):
    """the input of the map token embedding (B, M, NEMB): pos / 50, Fourier(pos), the unit direction of travel, count / 4 (0 = unknown), count known (count > 0); only the fields of the sample itself"""
    pos, cnt = maps["pos"].float(), maps["count"].float()
    return torch.cat([pos / HALF, JK.fourier(pos), maps["dir"].float(), (cnt / 4.0)[..., None], (cnt > 0).float()[..., None]], -1)


def map_geometry(pos, dirv, mpos, mdir, mcount):
    """pos / dirv (B, K, 2) the tokens, mpos / mdir (B, M, 2) and mcount (B, M) the map samples -> (B, K, M, NGEO), [b, i, j] = the sample j seen from the token i: cos and sin of the angle from the token heading to the sample's direction of travel (sin = cross(t_i, u_j)), the offset
    mpos_j - pos_i along the token heading and along its left normal (-t_y, t_x), both / 20, the distance / 20, the lane count / 4 and the flag count > 0"""
    ch = mpos[:, None, :, :] - pos[:, :, None, :]
    tq, tm = dirv[:, :, None, :].expand_as(ch), mdir[:, None, :, :].expand_as(ch)
    nq = torch.stack([-tq[..., 1], tq[..., 0]], -1)
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    cross = tq[..., 0] * tm[..., 1] - tq[..., 1] * tm[..., 0]
    cnt = mcount.to(ch.dtype)[:, None, :].expand(ch.shape[:3])
    return torch.stack([dot(tq, tm), cross, dot(ch, tq) / 20.0, dot(ch, nq) / 20.0, (dot(ch, ch) + 1e-12).sqrt() / 20.0, cnt / 4.0, (cnt > 0).to(ch.dtype)], -1)


class MapBlock(nn.Module):
    """the cross-attention block of the module docstring"""

    def __init__(self, d=D, heads=4, geo_hidden=32, emb_hidden=32):
        super().__init__()
        assert d % heads == 0, f"d {d} is not a multiple of heads {heads}"
        self.h, self.dh = heads, d // heads
        self.ln_q, self.ln_k = nn.LayerNorm(d), nn.LayerNorm(d)
        self.emb = nn.Sequential(nn.Linear(NEMB, emb_hidden), nn.GELU(), nn.Linear(emb_hidden, d))
        self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.k_none, self.v_none = nn.Parameter(torch.randn(heads, self.dh) * 0.1), nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.none_b = nn.Parameter(torch.zeros(heads))
        self.bias = nn.Sequential(nn.Linear(NGEO, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, heads))
        self.o = nn.Linear(d, d)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def attend(self, x, pos, dirv, valid, maps):
        """-> (att (B, h, K, M + 1) float32: the softmax over {the map samples} + {none} (the none key LAST; M = 0 without a map), v (B, h, M, dh) or None). The padding samples have probability exactly 0"""
        B, K, _ = x.shape
        q = self.q(self.ln_q(x)).view(B, K, self.h, self.dh).transpose(1, 2)                         # (B, h, K, dh)
        z_none = (q * self.k_none[None, :, None, :]).sum(-1).float() / math.sqrt(self.dh) + self.none_b[None, :, None]
        if not maps or maps["valid"].shape[1] == 0:
            return z_none[..., None].softmax(-1), None
        M = maps["valid"].shape[1]
        k, v = self.kv(self.ln_k(self.emb(map_features(maps)))).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)    # (B, h, M, dh) each
        g = map_geometry(pos, dirv, maps["pos"].to(pos.dtype), maps["dir"].to(pos.dtype), maps["count"])
        z = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh) + self.bias(g).float().permute(0, 3, 1, 2)
        z = z.masked_fill(~maps["valid"].bool()[:, None, None, :], NEG)
        return torch.cat([z, z_none[..., None]], -1).softmax(-1), v

    def delta(self, x, pos, dirv, valid, maps):
        """the residual update (B, K, D): the output projection of the attended values, zero on the padding queries"""
        B, K, D_ = x.shape
        att, v = self.attend(x, pos, dirv, valid, maps)
        vn = self.v_none[None, :, None, :].to(x.dtype)
        out = att[..., -1:].to(x.dtype) * vn if v is None else att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn
        return self.o(out.transpose(1, 2).reshape(B, K, D_)) * valid.to(x.dtype)[..., None]

    def forward(self, x, pos, dirv, valid, maps):
        return x + self.delta(x, pos, dirv, valid, maps)

    use = JP.PastBlock.use                                                                           # the kill measure: v (the map of another sample = `jpast.roll_past(maps)`), v_none, from the block's own update y = block(x, maps) - x


# ------------------------------------------------------------------------------------------------------------------------------ selftest (the mutants of test_jmap.py run it)
def _tokens(B, K, seed, n_invalid=0):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(B, K, generator=g) * 6.28
    valid = torch.ones(B, K, dtype=torch.bool)
    if n_invalid:
        valid[:, -n_invalid:] = False
    return torch.randn(B, K, D, generator=g), torch.randn(B, K, 2, generator=g) * 15, torch.stack([th.cos(), th.sin()], -1), valid


def _maps(B, M, seed, n_invalid=0, counts=(0, 1, 2, 3, 5)):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(B, M, generator=g) * 6.28
    valid = torch.ones(B, M, dtype=torch.bool)
    if n_invalid:
        valid[:, -n_invalid:] = False
    return dict(pos=torch.randn(B, M, 2, generator=g) * 15, dir=torch.stack([th.cos(), th.sin()], -1), count=torch.tensor(counts)[torch.randint(len(counts), (B, M), generator=g)], valid=valid)


def _trained(seed=0, **kw):
    torch.manual_seed(seed)
    b = MapBlock(**kw)
    for p in (b.o.weight, b.o.bias):
        nn.init.normal_(p, std=0.3)
    nn.init.normal_(b.none_b, std=0.5)
    return b


def reference(b, x, pos, dirv, valid, maps):
    """the block by loops over batch, head, query and key (plain float64 arithmetic of the module docstring, the features written out by hand): the reference of the selftest"""
    B, K, D_ = x.shape
    P = {n: p.detach().double() for n, p in b.named_parameters()}
    ln = lambda t, n: (t - t.mean(-1, keepdim=True)) / (t.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * P[n + ".weight"] + P[n + ".bias"]  # noqa: E731
    lin = lambda t, n: t @ P[n + ".weight"].T + P[n + ".bias"]  # noqa: E731
    out = x.double().clone()
    q = lin(ln(x.double(), "ln_q"), "q").view(B, K, b.h, b.dh)
    if maps:
        M = maps["valid"].shape[1]
        feats = []
        for bi in range(B):
            rows = []
            for j in range(M):
                p_, d_, c_ = maps["pos"][bi, j].double(), maps["dir"][bi, j].double(), float(maps["count"][bi, j])
                four = [torch.sin(p_[c] / HALF * math.pi * 2 ** k) for c in (0, 1) for k in range(JK.BANDS)]
                four2 = [torch.cos(p_[c] / HALF * math.pi * 2 ** k) for c in (0, 1) for k in range(JK.BANDS)]
                f = torch.stack([p_[0] / HALF, p_[1] / HALF] + four[:JK.BANDS] + four2[:JK.BANDS] + four[JK.BANDS:] + four2[JK.BANDS:] + [d_[0], d_[1], torch.tensor(c_ / 4.0, dtype=torch.float64), torch.tensor(float(c_ > 0), dtype=torch.float64)])
                rows.append(f)
            feats.append(torch.stack(rows))
        emb = torch.stack(feats)
        kv = lin(ln(lin(torch.nn.functional.gelu(lin(emb, "emb.0")), "emb.2"), "ln_k"), "kv").view(B, M, 2, b.h, b.dh)
    for bi in range(B):
        for i in range(K):
            if not bool(valid[bi, i]):
                continue
            att_out = torch.zeros(b.h, b.dh, dtype=torch.float64)
            for h in range(b.h):
                zs, vs = [], []
                if maps:
                    for j in range(M):
                        if not bool(maps["valid"][bi, j]):
                            continue
                        tq, tm = dirv[bi, i].double(), maps["dir"][bi, j].double()
                        ch = maps["pos"][bi, j].double() - pos[bi, i].double()
                        nq = torch.stack([-tq[1], tq[0]])
                        c_ = float(maps["count"][bi, j])
                        geo = torch.stack([tq @ tm, tq[0] * tm[1] - tq[1] * tm[0], ch @ tq / 20, ch @ nq / 20, ch.norm() / 20, torch.tensor(c_ / 4.0, dtype=torch.float64), torch.tensor(float(c_ > 0), dtype=torch.float64)])
                        zs.append(q[bi, i, h] @ kv[bi, j, 0, h] / math.sqrt(b.dh) + lin(torch.nn.functional.gelu(lin(geo, "bias.0")), "bias.2")[h])
                        vs.append(kv[bi, j, 1, h])
                zs.append(q[bi, i, h] @ P["k_none"][h] / math.sqrt(b.dh) + P["none_b"][h])
                vs.append(P["v_none"][h])
                a = torch.softmax(torch.stack(zs), 0)
                att_out[h] = (a[:, None] * torch.stack(vs)).sum(0)
            out[bi, i] = out[bi, i] + lin(att_out.reshape(-1), "o")
    return out


def selftest():
    torch.manual_seed(0)
    B, K, M = 2, 7, 6
    x, pos, dirv, valid = _tokens(B, K, 1, n_invalid=2)
    maps = _maps(B, M, 2, n_invalid=1)
    # the size: 20 264 parameters at d 64, 4 heads, 32 hidden in the embedding and the bias (a pinned number: a changed block is a new card)
    b0 = MapBlock()
    assert sum(p.numel() for p in b0.parameters()) == 20264, sum(p.numel() for p in b0.parameters())
    # zero-init = the identity, with and without a map
    for ma in (None, {}, maps):
        assert torch.equal(b0(x, pos, dirv, valid, ma), x), "the zero-initialised block is not the identity"
    # the loop reference of the whole block (a trained-like block, a map with a padding sample, unknown counts): features, embedding, norms, scale, bias, none key, masks, heads
    b = _trained()
    for ma in (None, {}, maps):
        got = b(x, pos, dirv, valid, ma).double()
        want = reference(b, x, pos, dirv, valid, ma)
        assert (got - want).abs().max() < 1e-4, ("reference", None if ma is None else sorted(ma), float((got - want).abs().max()))
    # no map = the none key only = an all-invalid map = the none value through the projection (by hand): the identity only while the projection is zero
    none = b(x, pos, dirv, valid, None)
    allpad = dict(maps, valid=torch.zeros_like(maps["valid"]))
    assert torch.equal(none, b(x, pos, dirv, valid, {})) and torch.equal(none, b(x, pos, dirv, valid, allpad)) and torch.equal(none, b(x, pos, dirv, valid, dict(maps, valid=maps["valid"][:, :0], pos=maps["pos"][:, :0], dir=maps["dir"][:, :0], count=maps["count"][:, :0]))), "no map is the all-invalid map"
    by_hand = x + torch.nn.functional.linear(b.v_none.reshape(-1), b.o.weight, b.o.bias)[None, None] * valid[..., None]
    assert torch.allclose(none, by_hand, atol=1e-6) and not torch.equal(none, x), "the none key only: the output is the none value through the projection"
    att, _ = b.attend(x, pos, dirv, valid, maps)
    assert att.shape == (B, 4, K, M + 1) and torch.allclose(att.sum(-1), torch.ones(B, 4, K), atol=1e-6) and (att[:, :, :, :-1][:, :, :, ~maps["valid"][0]] == 0).all() and (att[..., -1] > 0).all(), "softmax over the samples + none; a padding sample has probability 0"
    # padding samples are masked (changing a padded sample changes nothing), padding queries keep x
    junk = {k: v.clone() for k, v in maps.items()}
    pad = ~maps["valid"]
    junk["pos"][pad], junk["dir"][pad], junk["count"][pad] = 99.0, 0.5, 4
    assert torch.equal(b(x, pos, dirv, valid, maps), b(x, pos, dirv, valid, junk)), "a padding sample changed the output"
    assert torch.equal(b(x, pos, dirv, valid, maps)[~valid], x[~valid]), "a padding query was changed"
    # only the four fields are read: any other key of the dict (a lane-level field) changes nothing
    assert torch.equal(b(x, pos, dirv, valid, maps), b(x, pos, dirv, valid, dict(maps, frac=torch.rand(B, M), piece=torch.arange(M)[None].repeat(B, 1), adj=torch.ones(B, M, M)))), "a field beyond pos / dir / count / valid was read"
    # a permutation of the map samples (all fields together) leaves the output unchanged
    perm = torch.randperm(M)
    assert torch.allclose(b(x, pos, dirv, valid, maps), b(x, pos, dirv, valid, {k: v[:, perm] for k, v in maps.items()}), atol=1e-5), "not permutation invariant"
    # the relative geometry by hand: a token at the origin heading north, a sample at (3, 4) heading east with 3 lanes; a second token heading east; an unknown count; the flag
    g = map_geometry(torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]), torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), torch.tensor([[[3.0, 4.0], [3.0, 4.0]]]), torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]), torch.tensor([[3, 0]]))
    want = torch.tensor([[[[0.0, -1.0, 4 / 20, -3 / 20, 5 / 20, 0.75, 1.0], [0.0, -1.0, 4 / 20, -3 / 20, 5 / 20, 0.0, 0.0]], [[1.0, 0.0, 3 / 20, 4 / 20, 5 / 20, 0.75, 1.0], [1.0, 0.0, 3 / 20, 4 / 20, 5 / 20, 0.0, 0.0]]]])
    assert g.shape == (1, 2, 2, NGEO) and torch.allclose(g, want, atol=1e-6), (g, want)
    # the same five geometry features as the past block (`jpast.cross_geometry`: one convention for both blocks), on random inputs
    gj = JP.cross_geometry(pos, dirv, torch.zeros(B, K), maps["pos"], maps["dir"], torch.zeros(B, M))
    assert torch.allclose(map_geometry(pos, dirv, maps["pos"], maps["dir"], maps["count"])[..., :5], gj[..., :5], atol=1e-6), "the geometry differs from the past block's"
    # invariances of the geometry: a common translation and a common rotation of the tokens and the map (the features live in the token frame)
    gm = map_geometry(pos, dirv, maps["pos"], maps["dir"], maps["count"])
    sh = torch.tensor([13.0, -7.0])
    assert torch.allclose(gm, map_geometry(pos + sh, dirv, maps["pos"] + sh, maps["dir"], maps["count"]), atol=1e-4), "a common translation changed the features"
    th = torch.tensor(0.9)
    Rm = torch.stack([torch.stack([th.cos(), -th.sin()]), torch.stack([th.sin(), th.cos()])])
    assert torch.allclose(gm, map_geometry(pos @ Rm.T, dirv @ Rm.T, maps["pos"] @ Rm.T, maps["dir"] @ Rm.T, maps["count"]), atol=1e-4), "a common rotation changed the features"
    # the embedding input by hand: pos / 50, the Fourier code, the direction, count / 4, count known; an unknown count has both zero
    mf = map_features(dict(pos=torch.tensor([[[10.0, -5.0]]]), dir=torch.tensor([[[0.0, -1.0]]]), count=torch.tensor([[2]])))
    assert mf.shape == (1, 1, NEMB) and torch.allclose(mf[0, 0, :2], torch.tensor([0.2, -0.1])) and torch.allclose(mf[0, 0, 2:18], JK.fourier(torch.tensor([[[10.0, -5.0]]]))[0, 0]) and torch.allclose(mf[0, 0, 18:], torch.tensor([0.0, -1.0, 0.5, 1.0]))
    assert torch.allclose(map_features(dict(pos=torch.zeros(1, 1, 2), dir=torch.zeros(1, 1, 2), count=torch.tensor([[0]])))[0, 0, -2:], torch.zeros(2))
    # the count is read (a trained block's output depends on it), the direction too
    assert not torch.equal(b(x, pos, dirv, valid, maps), b(x, pos, dirv, valid, dict(maps, count=torch.full_like(maps["count"], 2)))) and not torch.equal(b(x, pos, dirv, valid, maps), b(x, pos, dirv, valid, dict(maps, dir=-maps["dir"])))
    # gradient: at zero init only the projection has one; with a non-zero projection every parameter has one (the embedding too), the tokens' states, and no constant of the map
    for p in b0.parameters():
        p.grad = None
    xg = x.clone().requires_grad_()
    b0(xg, pos, dirv, valid, maps).square().sum().backward()
    assert b0.o.weight.grad.abs().sum() > 0 and all(p.grad is None or p.grad.abs().sum() == 0 for n, p in b0.named_parameters() if not n.startswith("o.")), "at zero init only the projection learns"
    bg = _trained(2)
    xg = x.clone().requires_grad_()
    mg = {k: (v.clone().float().requires_grad_() if k in ("pos", "dir") else v) for k, v in maps.items()}
    bg(xg, pos, dirv, valid, mg).square().sum().backward()
    dead = [n for n, p in bg.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not dead and xg.grad.abs().sum() > 0, f"no gradient: {dead}"
    assert all(mg[k].grad is not None and mg[k].grad.abs().sum() > 0 for k in ("pos", "dir")), "the map input is differentiable (a constant of the data, but the path exists)"
    # u, the kill measures (module docstring): v_none (the none key only) is 0 at zero init; v (the map of ANOTHER SAMPLE) needs B >= 2 or a donor
    y0 = b0(x, pos, dirv, valid, maps) - x
    u0 = b0.use(x, pos, dirv, valid, y0, maps)
    assert u0["d2_none"] == 0.0 and u0["d2_other"] == 0.0 and u0["n_other"] == int(valid.sum()) and u0["n"] == int(valid.sum()), "at zero init the block changes nothing"
    y = b(x, pos, dirv, valid, maps) - x
    su = b.use(x, pos, dirv, valid, y, maps)
    y_none = b(x, pos, dirv, valid, None) - x
    y_oth = b(x, pos, dirv, valid, JP.roll_past(maps)) - x                                                   # the map of ANOTHER SAMPLE: every tensor rolled by one along the batch axis
    mm = valid[..., None].double()
    d2n, d2o = (float((((y - r_).double() * mm) ** 2).sum()) for r_ in (y_none, y_oth))
    assert abs(su["d2_none"] - d2n) < 1e-6 * max(d2n, 1) and abs(su["d2_other"] - d2o) < 1e-6 * max(d2o, 1) and su["n"] == su["n_other"] == int(valid.sum()) and d2n > 0 and d2o > 0, "use: the sums by hand"
    rm = JP.roll_past(maps)
    assert all(torch.equal(rm[k][0], maps[k][1]) and torch.equal(rm[k][1], maps[k][0]) for k in maps) and not torch.equal(rm["pos"], maps["pos"]), "the other sample's map: every tensor rolled by one"
    one = {k: v[:1] for k, v in maps.items()}                                                                # a batch of ONE sample: no other sample, n_other 0; with a donor (sample 1's map) = what the batch gave sample 0
    y_one = b(x[:1], pos[:1], dirv[:1], valid[:1], one) - x[:1]
    s1 = b.use(x[:1], pos[:1], dirv[:1], valid[:1], y_one, one)
    assert s1["n_other"] == 0 and s1["d2_other"] == 0.0 and s1["d2_none"] > 0 and s1["n"] == int(valid[0].sum())
    s2 = b.use(x[:1], pos[:1], dirv[:1], valid[:1], y_one, one, {k: v[1:2] for k, v in maps.items()})
    assert s2["n_other"] == int(valid[0].sum()) and abs(s2["d2_other"] - float((((y - y_oth)[:1].double() * mm[:1]) ** 2).sum())) < 1e-6 * max(d2o, 1), "a donor is the other sample"
    # the critic's case by hand: a block that ignores the CONTENT of the map. q = 0, geometry bias 0, v = the constant 1 for every key, v_none = 0, o = identity: with n valid samples every valid query token gets y = t on all D channels, t = n / (n + 1)
    # (the logits are all 0: the attention is uniform over the n samples and the none key); v_none (the none key only) = t / RMS(x) > 0, v (the other sample's map) = 0. A block that READS the map (v = the embedding, LayerNormed) has v > 0
    def plain(content):
        blk = MapBlock()
        with torch.no_grad():
            for p_ in (blk.q.weight, blk.q.bias, blk.bias[2].weight, blk.bias[2].bias, blk.v_none, blk.kv.weight):
                p_.zero_()
            blk.kv.bias.zero_()
            blk.o.weight.copy_(torch.eye(D))
            if content:
                blk.kv.weight[D:].copy_(torch.eye(D))
            else:
                blk.kv.bias[D:] = 1.0
        return blk
    xq, pq, dq, vq = _tokens(2, 7, 11, n_invalid=2)                                                         # 5 valid query tokens per sample
    mq = _maps(2, 6, 12, n_invalid=2)                                                                         # 4 valid map samples per sample, different contents
    n_ = 4
    t_ = n_ / (n_ + 1)
    x2q = float((xq.double()[vq] ** 2).sum())
    blind = plain(False)
    sb = blind.use(xq, pq, dq, vq, blind(xq, pq, dq, vq, mq) - xq, mq)
    assert abs(sb["d2_none"] - 10 * D * t_ ** 2) < 1e-5 * 10 * D * t_ ** 2 and sb["d2_other"] < 1e-9 and sb["n"] == sb["n_other"] == 10 and abs(math.sqrt(sb["d2_none"] / sb["x2"]) - t_ / math.sqrt(x2q / (10 * D))) < 1e-5, f"a block that ignores the map content: {sb}"
    reader = plain(True)
    sr = reader.use(xq, pq, dq, vq, reader(xq, pq, dq, vq, mq) - xq, mq)
    assert sr["d2_other"] > 1.0 and sr["d2_none"] > 1.0, f"a block that reads the map content: {sr}"
    um = UseMeter(blind)                                                                                     # the hook on these blocks: u = 0 and u_none > 0 for the blind one; a batch of one needs a donor
    blind(xq, pq, dq, vq, mq)
    tk = um.take()
    assert tk["u"] < 1e-6 and abs(tk["u_none"] - math.sqrt(sb["d2_none"] / sb["x2"])) < 1e-9 and tk["n"] == tk["n_other"] == 10 and um.take() is None
    blind(xq[:1], pq[:1], dq[:1], vq[:1], {k: v[:1] for k, v in mq.items()})
    tk = um.take()
    assert tk["u"] is None and tk["u_none"] > 0 and tk["n_other"] == 0 and tk["n"] == 5, "a batch of one sample without a donor has no v"
    um.detach()
    um = UseMeter(reader)
    um.donor = {k: v[1:2] for k, v in mq.items()}
    reader(xq[:1], pq[:1], dq[:1], vq[:1], {k: v[:1] for k, v in mq.items()})
    tk = um.take()
    assert tk["u"] > 0.05 and tk["n_other"] == 5, "a donor for a batch of one"
    um.detach()
    um = UseMeter(b)                                                                                         # calls without a map ({}) are not counted
    b(x, pos, dirv, valid, None), b(x, pos, dirv, valid, {})
    assert um.take() is None
    um.detach()
    # the real head with the hook: the zero block = the plain head bit for bit, a trained block changes it, the loss reaches every parameter of the block, `{}` = an all-invalid map = no map
    hb = JK.synthetic_batch(2, 48, seed=3)
    torch.manual_seed(7)
    plain_head = JK.JTokenHead().eval()
    torch.manual_seed(7)
    head = JK.JTokenHead(map_block=MapBlock(d=D, heads=4)).eval()
    head.load_state_dict(dict(plain_head.state_dict(), **{f"map_block.{k}": v for k, v in head.map_block.state_dict().items()}))
    mh = _maps(2, 20, 5, n_invalid=3)
    with torch.no_grad():
        o_plain, o_zero = plain_head(hb), head(hb, maps=mh)
        assert all(torch.equal(o_plain[k], o_zero[k]) for k in ("logp", "types", "x", "w")), "the zero-initialised block changed the head"
        head.map_block.load_state_dict(_trained(4).state_dict())
        o_live, o_none, o_empty = head(hb, maps=mh), head(hb, maps=dict(mh, valid=torch.zeros_like(mh["valid"]))), head(hb, maps={})
        assert not torch.equal(o_live["x"], o_plain["x"]) and torch.equal(o_none["logp"], o_empty["logp"]) and not torch.equal(o_none["logp"], o_live["logp"]), "the hook: a trained block changes the head, {} = an all-invalid map"
        assert torch.equal(head(hb)["logp"], o_plain["logp"]), "maps None: the block is not called (a trained block, the plain head's output)"
    head.train()
    head.map_block.load_state_dict(_trained(5).state_dict())
    head.zero_grad()
    JK.jtoken_loss(head(hb, maps=mh), hb)[0].backward()
    dead = [n for n, p in head.map_block.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not dead, f"the loss does not reach the block: {dead}"
    # a padding-only batch row and K = 512 tokens x M = 704 map samples (the cap of `lane_map_data`; memory below 3 GB on the CPU, bf16 autocast finite)
    bb = _trained(3)
    xb, pb, db, vb = _tokens(2, 512, 4, n_invalid=100)
    mb = _maps(2, 704, 5, n_invalid=50)
    xb = xb.clone().requires_grad_()
    ob = bb(xb, pb, db, vb, mb)
    ob.square().mean().backward()
    assert ob.shape == (2, 512, D) and torch.isfinite(ob).all() and torch.isfinite(xb.grad).all()
    import resource
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 < 3072, "K = 512, M = 704, batch 2, forward and backward: more than 3 GB"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(bb(xb.detach(), pb, db, vb, mb)).all()
    print("jmap selftest ok")


if __name__ == "__main__":
    assert len(sys.argv) > 1 and sys.argv[1] == "selftest", "usage: jmap.py selftest"
    selftest()
