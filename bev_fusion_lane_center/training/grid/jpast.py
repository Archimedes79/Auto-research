"""E368 (stage S2 of the token champion, D113): ONE cross-attention block from the tokens of the current keyframe to the token STATES of the previous keyframe, in the lane frame of the current one. The head of E365 (`jtoken.JTokenHead`) calls it after self-attention layer 1.

    block = PastBlock(d=64, heads=4)                          # 17.4 k parameters; the output projection starts at ZERO: the block is the identity at the start (a trained head can be extended)
    head.past_block = block                                   # the hook: x = block(x, pos, dirv, valid, past) after layer 1 when forward(batch, past) gets a past (an empty dict = no past)
    past = make_past(out_prev, frame_prev, frame_cur, rel_R, rel_t, learned)      # the state of step k - 1 as the block reads it; {} (or None) = a first frame: the none key only
    pos2, dirv2 = warp_tokens(pos, dirv, rel_R, rel_t)       # p_cur = R p_prev + t (the pose of `data/lane_past_data.py`)
    g = cross_geometry(pos_q, dirv_q, e_q, pos_k, dirv_k, e_k)  # (B, Kq, Kk, 7) relative geometry of every (query, key) pair

PastBlock.forward(x (B, K, D), pos, dirv (B, K, 2) the current tokens, valid (B, K) bool, past) -> x (B, K, D). past = {} / None or dict(x (B, Kp, D) the previous token states, pos / dirv (B, Kp, 2) ALREADY warped into this lane frame, valid (B, Kp) bool,
e (B, Kp) the existence probability of the previous tokens, e_cur (B, K) that of the current tokens (the head's hook passes no e, so it travels here), w (B, Kp) the key weights or None). Pre-norm: LayerNorm of x for the queries, of the past states for keys and
values (the state is the head's own residual stream: its scale is the head's). Per head: logit = q.k / sqrt(dh) + bias_h(cross_geometry) + log(w_j + 1e-3) for the past keys (padding keys masked) and ONE learned none key (k_none, v_none and an own bias) per head;
softmax over {the past keys} + {none}; the heads are concatenated and projected (zero-initialised); residual add; padding queries (valid False) keep x. A first frame / an all-invalid past: the softmax has the none key only (the output is the none value
through the projection). The 7 features of cross_geometry: cos and sin of the turn from the query heading to the key heading, the offset pos_k - pos_q along / across the QUERY heading (across = its left normal) / 20, the distance / 20, e_q, e_k
(the card says 8 and names these 7).
`make_past`: the key weight w of the previous tokens is sigmoid(e) of the cached existence while `learned` is False (the first `jtoken.SWITCH` training steps: the head's own read-out is still untrained) and the previous step's own 1 - p_none (`out_prev['w']`) after
it; the state x and w are NOT detached (the gradient runs through the steps); only the input keys of the frames are read (pos, dirv, valid, e): nothing of the GT enters the state. `PastBlock.use` is the kill measure of the card: u = RMS(block(x, past) - block(x, none
key only)) / RMS(x) on the valid tokens. Tests: grid/test_jpast.py (hand-computed values, a loop reference of the block, the invariances, the mutants)
"""
import math

import torch
import torch.nn as nn

D = 64
NGEO = 7
NEG = -1e4                                                                  # the fill of a masked logit (as `jtoken.NEG`)


def warp_tokens(pos, dirv, rel_R, rel_t):
    """pos, dirv (B, K, 2), rel_R (B, 2, 2), rel_t (B, 2) -> the tokens of the previous keyframe in the lane frame of the current one: pos' = R pos + t (the row form pos R^T + t), dirv' = R dirv (a unit vector stays a unit vector)"""
    R = rel_R.to(pos.dtype)
    return torch.einsum("bij,bkj->bki", R, pos) + rel_t.to(pos.dtype)[:, None], torch.einsum("bij,bkj->bki", R, dirv.to(pos.dtype))


def cross_geometry(pos_q, dirv_q, e_q, pos_k, dirv_k, e_k):
    """pos / dirv (B, Kq, 2) and (B, Kk, 2), e (B, Kq) and (B, Kk) probabilities -> (B, Kq, Kk, NGEO), [b, i, j] = the key j seen from the query i: cos and sin of the turn from the query heading to the key heading (sin = cross(t_q, t_k), the sign of
    `jtok.pair_geometry`), the offset pos_k - pos_q along and across the query heading (across = the left normal (-t_y, t_x)) / 20, the distance / 20, e_q, e_k"""
    ch = pos_k[:, None, :, :] - pos_q[:, :, None, :]
    tq, tk = dirv_q[:, :, None, :].expand_as(ch), dirv_k[:, None, :, :].expand_as(ch)
    nq = torch.stack([-tq[..., 1], tq[..., 0]], -1)
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    cross = tq[..., 0] * tk[..., 1] - tq[..., 1] * tk[..., 0]
    return torch.stack([dot(tq, tk), cross, dot(ch, tq) / 20.0, dot(ch, nq) / 20.0, (dot(ch, ch) + 1e-12).sqrt() / 20.0, e_q[:, :, None].expand(ch.shape[:3]), e_k[:, None, :].expand(ch.shape[:3])], -1)


def _const(t):
    return t.detach()


def make_past(out_prev, frame_prev, frame_cur, rel_R, rel_t, learned, has_prev=None):
    """the state of step k - 1 for step k: out_prev = the head's output dict of the previous frame (x (B, K, D), w (B, K) = 1 - p_none), frame_prev / frame_cur = their batch dicts (only pos, dirv, valid, e are read), rel_R (B, 2, 2) / rel_t (B, 2) = the pose of
    the previous frame in the lane frame of the current one, learned = the head's key-weight switch is over, has_prev (B,) bool = the sample has a previous keyframe (None: all do; a sample without one gets an all-invalid past = the none key only) -> the dict
    `PastBlock` reads. x and w keep their graph (no detach); the cached inputs are constants"""
    pos, dirv = warp_tokens(_const(frame_prev["pos"]).float(), _const(frame_prev["dirv"]).float(), rel_R.float(), rel_t.float())
    e = _const(frame_prev["e"]).float().sigmoid()
    valid = _const(frame_prev["valid"]).bool()
    if has_prev is not None:
        valid = valid & has_prev.bool()[:, None]
    return dict(x=out_prev["x"], pos=pos, dirv=dirv, valid=valid, e=e, e_cur=_const(frame_cur["e"]).float().sigmoid(), w=out_prev["w"] if learned else e)


class UseMeter:
    """the kill measure u of the card as a forward hook on a `PastBlock` (the head calls it as block(x, pos, dirv, valid, past)): every call with a non-empty past adds (sum (y - y_none)^2, sum x^2, valid tokens); `take()` -> (u, valid tokens) of the calls since the
    last `take` (None without a call) and clears (`take_sums`: the three sums); `detach()` removes the hook"""

    def __init__(self, block):
        self.acc, self.handle = [0.0, 0.0, 0], block.register_forward_hook(self._hook)

    def _hook(self, block, args, out):
        x, pos, dirv, valid, past = args
        if not past:
            return
        for i, v in enumerate(block.use(x.detach(), pos, dirv, valid, (out - x).detach())):
            self.acc[i] += v

    def take_sums(self):
        s, self.acc = self.acc, [0.0, 0.0, 0]
        return tuple(s)

    def take(self):
        d2, x2, n = self.take_sums()
        return (math.sqrt(d2 / max(x2, 1e-30)), n) if n else None

    def detach(self):
        self.handle.remove()


class PastBlock(nn.Module):
    """the cross-attention block of the module docstring"""

    def __init__(self, d=D, heads=4, geo_hidden=32):
        super().__init__()
        assert d % heads == 0, f"d {d} is not a multiple of heads {heads}"
        self.h, self.dh = heads, d // heads
        self.ln_q, self.ln_k = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv = nn.Linear(d, d), nn.Linear(d, 2 * d)
        self.k_none, self.v_none = nn.Parameter(torch.randn(heads, self.dh) * 0.1), nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.none_b = nn.Parameter(torch.zeros(heads))
        self.bias = nn.Sequential(nn.Linear(NGEO, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, heads))
        self.o = nn.Linear(d, d)
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def attend(self, x, pos, dirv, valid, past):
        """-> (att (B, h, K, Kp + 1) float32: the softmax over {the past keys} + {none} (the none key LAST; Kp = 0 without a past), v (B, h, Kp, dh) or None). The padding keys have probability exactly 0"""
        B, K, _ = x.shape
        q = self.q(self.ln_q(x)).view(B, K, self.h, self.dh).transpose(1, 2)                         # (B, h, K, dh)
        z_none = (q * self.k_none[None, :, None, :]).sum(-1).float() / math.sqrt(self.dh) + self.none_b[None, :, None]
        if not past:
            return z_none[..., None].softmax(-1), None
        Kp = past["x"].shape[1]
        k, v = self.kv(self.ln_k(past["x"])).view(B, Kp, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)    # (B, h, Kp, dh) each
        g = cross_geometry(pos, dirv, past["e_cur"], past["pos"], past["dirv"], past["e"])
        z = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh) + self.bias(g).float().permute(0, 3, 1, 2)
        if past.get("w") is not None:
            z = z + (past["w"].float() + 1e-3).log()[:, None, None, :]
        z = z.masked_fill(~past["valid"].bool()[:, None, None, :], NEG)
        return torch.cat([z, z_none[..., None]], -1).softmax(-1), v

    def delta(self, x, pos, dirv, valid, past):
        """the residual update (B, K, D): the output projection of the attended values, zero on the padding queries"""
        B, K, D_ = x.shape
        att, v = self.attend(x, pos, dirv, valid, past)
        vn = self.v_none[None, :, None, :].to(x.dtype)
        out = att[..., -1:].to(x.dtype) * vn if v is None else att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn
        return self.o(out.transpose(1, 2).reshape(B, K, D_)) * valid.to(x.dtype)[..., None]

    def forward(self, x, pos, dirv, valid, past):
        return x + self.delta(x, pos, dirv, valid, past)

    @torch.no_grad()
    def use(self, x, pos, dirv, valid, y):
        """the kill measure of the card from the block's own update y = block(x, past) - x (as the forward made it): -> (sum of (y - y_none)^2, sum of x^2, valid tokens) over the valid tokens and the D channels, y_none = the update with the none key only;
        u = sqrt(sum1 / sum2) = RMS(y - y_none) / RMS(x), summed over batches before the ratio"""
        m = valid.to(x.dtype)[..., None]
        y0 = self.delta(x, pos, dirv, valid, None)
        return float((((y - y0) * m) ** 2).sum()), float(((x * m) ** 2).sum()), int(valid.sum())


# ------------------------------------------------------------------------------------------------------------------------------ selftest (the mutants of test_jpast.py run it)
def _frame(B, K, seed, n_invalid=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    th = torch.rand(B, K, generator=g) * 6.28
    valid = torch.ones(B, K, dtype=torch.bool)
    if n_invalid:
        valid[:, -n_invalid:] = False
    return dict(x=r(B, K, D), pos=r(B, K, 2) * 15, dirv=torch.stack([th.cos(), th.sin()], -1), valid=valid, e=torch.rand(B, K, generator=g), w=torch.rand(B, K, generator=g))


def _past_of(f, e_cur):
    return dict(x=f["x"], pos=f["pos"], dirv=f["dirv"], valid=f["valid"], e=f["e"], w=f["w"], e_cur=e_cur)


def _trained(seed=0, **kw):
    torch.manual_seed(seed)
    b = PastBlock(**kw)
    for p in (b.o.weight, b.o.bias):
        nn.init.normal_(p, std=0.3)
    nn.init.normal_(b.none_b, std=0.5)
    return b


def reference(b, x, pos, dirv, valid, past):
    """the block by loops over batch, head, query and key (plain float64 arithmetic of the module docstring): the reference of the selftest"""
    B, K, D_ = x.shape
    P = {n: p.detach().double() for n, p in b.named_parameters()}
    ln = lambda t, n: (t - t.mean(-1, keepdim=True)) / (t.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * P[n + ".weight"] + P[n + ".bias"]  # noqa: E731
    lin = lambda t, n: t @ P[n + ".weight"].T + P[n + ".bias"]  # noqa: E731
    out = x.double().clone()
    q = lin(ln(x.double(), "ln_q"), "q").view(B, K, b.h, b.dh)
    if past:
        Kp = past["x"].shape[1]
        kv = lin(ln(past["x"].double(), "ln_k"), "kv").view(B, Kp, 2, b.h, b.dh)
    for bi in range(B):
        for i in range(K):
            if not bool(valid[bi, i]):
                continue
            att_out = torch.zeros(b.h, b.dh, dtype=torch.float64)
            for h in range(b.h):
                zs, vs = [], []
                if past:
                    for j in range(Kp):
                        if not bool(past["valid"][bi, j]):
                            continue
                        tq, tk = dirv[bi, i].double(), past["dirv"][bi, j].double()
                        ch = past["pos"][bi, j].double() - pos[bi, i].double()
                        nq = torch.stack([-tq[1], tq[0]])
                        geo = torch.stack([tq @ tk, tq[0] * tk[1] - tq[1] * tk[0], ch @ tq / 20, ch @ nq / 20, ch.norm() / 20, past["e_cur"][bi, i].double(), past["e"][bi, j].double()])
                        bias = lin(torch.nn.functional.gelu(lin(geo, "bias.0")), "bias.2")[h]
                        z = q[bi, i, h] @ kv[bi, j, 0, h] / math.sqrt(b.dh) + bias
                        if past.get("w") is not None:
                            z = z + math.log(float(past["w"][bi, j]) + 1e-3)
                        zs.append(z)
                        vs.append(kv[bi, j, 1, h])
                zs.append(q[bi, i, h] @ P["k_none"][h] / math.sqrt(b.dh) + P["none_b"][h])
                vs.append(P["v_none"][h])
                a = torch.softmax(torch.stack(zs), 0)
                att_out[h] = (a[:, None] * torch.stack(vs)).sum(0)
            out[bi, i] = out[bi, i] + lin(att_out.reshape(-1), "o")
    return out


def selftest():
    torch.manual_seed(0)
    B, K, Kp = 2, 7, 6
    f, fp = _frame(B, K, 1, n_invalid=2), _frame(B, Kp, 2, n_invalid=1)
    e_cur = f["e"]
    past = _past_of(fp, e_cur)
    # the size: 17 416 parameters at d 64, 4 heads, 32 hidden (a pinned number: a changed block is a new card)
    b0 = PastBlock()
    assert sum(p.numel() for p in b0.parameters()) == 17416, sum(p.numel() for p in b0.parameters())
    # zero-init = the identity, with and without a past
    for pa in (None, {}, past):
        assert torch.equal(b0(f["x"], f["pos"], f["dirv"], f["valid"], pa), f["x"]), "the zero-initialised block is not the identity"
    # the loop reference of the whole block (a trained-like block, a past with a padding key, key weights): scale, bias, none key, masks, heads, norms
    b = _trained()
    for pa in (None, past, dict(past, w=None)):
        got = b(f["x"], f["pos"], f["dirv"], f["valid"], pa).double()
        want = reference(b, f["x"], f["pos"], f["dirv"], f["valid"], pa)
        assert (got - want).abs().max() < 1e-4, ("reference", None if pa is None else sorted(pa), float((got - want).abs().max()))
    # a first frame = the none key only = an all-invalid past = the none value through the projection (by hand)
    x, pos, dirv, valid = f["x"], f["pos"], f["dirv"], f["valid"]
    first = b(x, pos, dirv, valid, None)
    assert torch.equal(first, b(x, pos, dirv, valid, {})) and torch.equal(first, b(x, pos, dirv, valid, dict(past, valid=torch.zeros_like(past["valid"])))), "a first frame is not the all-invalid past"
    by_hand = x + torch.nn.functional.linear(b.v_none.reshape(-1), b.o.weight, b.o.bias)[None, None] * valid[..., None]
    assert torch.allclose(first, by_hand, atol=1e-6), "the none key only: the output is the none value through the projection"
    att, _ = b.attend(x, pos, dirv, valid, past)
    assert att.shape == (B, 4, K, Kp + 1) and torch.allclose(att.sum(-1), torch.ones(B, 4, K), atol=1e-6) and (att[:, :, :, :-1][:, :, :, ~past["valid"][0]] == 0).all(), "softmax over the keys + none; a padding key has probability 0"
    assert (att[..., -1] > 0).all()
    # padding keys are masked (changing a padded past token changes nothing), padding queries keep x, the padding values of the current frame do not leak either
    junk = dict(past, x=past["x"].clone(), pos=past["pos"].clone(), dirv=past["dirv"].clone(), e=past["e"].clone(), w=past["w"].clone())
    pad = ~past["valid"]
    junk["x"][pad], junk["pos"][pad], junk["e"][pad], junk["w"][pad] = 9.0, 99.0, 0.77, 5.0
    assert torch.equal(b(x, pos, dirv, valid, past), b(x, pos, dirv, valid, junk)), "a padding key changed the output"
    assert torch.equal(b(x, pos, dirv, valid, past)[~valid], x[~valid]), "a padding query was changed"
    # a permutation of the past tokens (all fields together) leaves the output unchanged
    perm = torch.randperm(Kp)
    assert torch.allclose(b(x, pos, dirv, valid, past), b(x, pos, dirv, valid, {k: v[:, perm] for k, v in past.items() if k != "e_cur"} | {"e_cur": e_cur}), atol=1e-5), "not permutation invariant"
    # invariances of the geometry: a common translation and a common rotation of the current and the past tokens (the features live in the query frame)
    sh = torch.tensor([13.0, -7.0])
    assert torch.allclose(b(x, pos, dirv, valid, past), b(x, pos + sh, dirv, valid, dict(past, pos=past["pos"] + sh)), atol=1e-4), "a common translation changed the output"
    th = torch.tensor(0.9)
    Rm = torch.stack([torch.stack([th.cos(), -th.sin()]), torch.stack([th.sin(), th.cos()])])
    rot = lambda t: t @ Rm.T  # noqa: E731
    assert torch.allclose(b(x, pos, dirv, valid, past), b(x, rot(pos), rot(dirv), valid, dict(past, pos=rot(past["pos"]), dirv=rot(past["dirv"]))), atol=1e-4), "a common rotation changed the output"
    # the key weight: log(w + 1e-3) is a bias of the logit: two identical keys with w 1 and 0.5 (the geometry bias zeroed) have the attention ratio (1.001) / (0.501); w -> 0 takes a key away
    b2 = _trained(1)
    nn.init.zeros_(b2.bias[2].weight)
    nn.init.zeros_(b2.bias[2].bias)
    one = dict(x=torch.randn(1, 1, D).repeat(1, 2, 1), pos=torch.zeros(1, 2, 2), dirv=torch.tensor([[[0.0, 1.0]] * 2]), valid=torch.ones(1, 2, dtype=torch.bool), e=torch.full((1, 2), 0.5), e_cur=torch.full((1, 3), 0.5))
    cur = (torch.randn(1, 3, D), torch.zeros(1, 3, 2), torch.tensor([[[0.0, 1.0]] * 3]), torch.ones(1, 3, dtype=torch.bool))
    for w, ratio in ((torch.tensor([[1.0, 0.5]]), 1.001 / 0.501), (torch.tensor([[1.0, 0.0]]), 1.001 / 0.001)):
        a, _ = b2.attend(*cur, dict(one, w=w))
        assert torch.allclose(a[0, :, 0, 0] / a[0, :, 0, 1], torch.full((4,), ratio), rtol=1e-4), ("the key weight is log(w + 1e-3)", a[0, :, 0, :2], ratio)
    a, _ = b2.attend(*cur, dict(one, w=None))
    assert torch.allclose(a[0, :, 0, 0], a[0, :, 0, 1]), "w None: no key weight"
    # gradient: at zero init only the projection has one; with a non-zero projection every parameter has one, the past state's only when it is not detached
    for p in b0.parameters():
        p.grad = None
    xg = f["x"].clone().requires_grad_()
    pg = dict(past, x=past["x"].clone().requires_grad_())
    b0(xg, pos, dirv, valid, pg).square().sum().backward()
    assert b0.o.weight.grad.abs().sum() > 0 and all(p.grad is None or p.grad.abs().sum() == 0 for n, p in b0.named_parameters() if not n.startswith("o.")), "at zero init only the projection learns"
    bg = _trained(2)
    xg = f["x"].clone().requires_grad_()
    pg = dict(past, x=past["x"].clone().requires_grad_())
    bg(xg, pos, dirv, valid, pg).square().sum().backward()
    dead = [n for n, p in bg.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    assert not dead and xg.grad.abs().sum() > 0 and pg["x"].grad.abs().sum() > 0, f"no gradient: {dead}"
    pd = dict(past, x=past["x"].clone().requires_grad_())
    pd2 = dict(pd, x=pd["x"].detach())
    xg = f["x"].clone().requires_grad_()
    bg.zero_grad()
    bg(xg, pos, dirv, valid, pd2).square().sum().backward()
    assert pd["x"].grad is None and bg.kv.weight.grad.abs().sum() > 0, "a detached past state got a gradient"
    # warp by hand: a 90 degree rotation and a translation; the unit headings stay unit, a heading is not translated; the pose of lane_past_data (p_cur = R p_prev + t)
    R90 = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])
    p2, d2 = warp_tokens(torch.tensor([[[1.0, 0.0], [0.0, 2.0]]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), R90, torch.tensor([[2.0, 3.0]]))
    assert torch.allclose(p2, torch.tensor([[[2.0, 4.0], [0.0, 3.0]]])) and torch.allclose(d2, torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])), (p2, d2)
    assert torch.allclose(warp_tokens(f["pos"], f["dirv"], torch.eye(2)[None].repeat(B, 1, 1), torch.zeros(B, 2))[0], f["pos"]) and torch.allclose(warp_tokens(pos, dirv, R90.repeat(B, 1, 1), torch.zeros(B, 2))[1].norm(dim=-1), torch.ones(B, K), atol=1e-6)
    pw = pos.clone().requires_grad_()
    warp_tokens(pw, dirv, R90.repeat(B, 1, 1), torch.ones(B, 2))[0].sum().backward()
    assert torch.allclose(pw.grad, torch.tensor([1.0, -1.0]).expand_as(pw)), "the warp is differentiable: d sum(R p + t) / dp = the column sums of R"
    # cross_geometry by hand: the query at the origin heading north, the key at (3, 4) heading east; a second query heading east
    g = cross_geometry(torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]), torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), torch.tensor([[0.2, 0.3]]), torch.tensor([[[3.0, 4.0]]]), torch.tensor([[[1.0, 0.0]]]), torch.tensor([[0.9]]))
    want = torch.tensor([[[[0.0, -1.0, 4 / 20, -3 / 20, 5 / 20, 0.2, 0.9]], [[1.0, 0.0, 3 / 20, 4 / 20, 5 / 20, 0.3, 0.9]]]])
    assert g.shape == (1, 2, 1, NGEO) and torch.allclose(g, want, atol=1e-6), (g, want)
    # make_past: the key weight is sigmoid(e) before the switch and the previous step's own w after it; no GT key; no detach; the pose moves the tokens
    out_prev = dict(x=torch.randn(B, K, D, requires_grad=True), w=torch.rand(B, K))
    fr = dict(pos=pos, dirv=dirv, valid=valid, e=torch.randn(B, K))
    gt = dict(y=torch.ones(B, K, K), ylong=torch.ones(B, K, K), yscale=torch.ones(B, K, K, 2), ytype=torch.ones(B, K, dtype=torch.long))
    Rb, tb = R90.repeat(B, 1, 1), torch.tensor([[2.0, 3.0]]).repeat(B, 1)
    m0, m1 = make_past(out_prev, fr, dict(fr, e=fr["e"] + 1), Rb, tb, False), make_past(out_prev, dict(fr, **gt), fr, Rb, tb, True)
    assert torch.equal(m0["w"], fr["e"].sigmoid()) and torch.equal(m1["w"], out_prev["w"]) and m0["x"] is out_prev["x"] and torch.equal(m0["e"], fr["e"].sigmoid()) and torch.equal(m0["e_cur"], (fr["e"] + 1).sigmoid())
    assert torch.allclose(m0["pos"], warp_tokens(pos, dirv, Rb, tb)[0]) and torch.equal(m0["valid"], valid) and sorted(m0) == ["dirv", "e", "e_cur", "pos", "valid", "w", "x"]
    assert all(torch.equal(m0[k], make_past(out_prev, dict(fr, **gt), fr, Rb, tb, False)[k]) for k in ("pos", "dirv", "e", "w")), "a GT key changed the state"
    m0["x"].sum().backward()
    assert out_prev["x"].grad is not None, "make_past detached the state"
    mh = make_past(out_prev, fr, fr, Rb, tb, False, torch.tensor([True, False]))                         # a sample without a previous keyframe gets the all-invalid past = the none key only
    assert torch.equal(mh["valid"][0], valid[0]) and not mh["valid"][1].any() and torch.equal(b(x, pos, dirv, valid, mh)[1], b(x, pos, dirv, valid, None)[1]) and not torch.equal(b(x, pos, dirv, valid, mh)[0], b(x, pos, dirv, valid, None)[0])
    assert torch.equal(make_past(out_prev, fr, fr, Rb, tb, False)["valid"], valid) and torch.equal(m0["valid"], valid), "has_prev None: every sample has a past"
    # u, the kill measure: 0 without a past (the none key only), 0 at zero init, > 0 with a past and a trained block, = the RMS ratio by hand
    ysub = lambda blk, pa: blk(x, pos, dirv, valid, pa) - x  # noqa: E731
    assert b0.use(x, pos, dirv, valid, ysub(b0, past))[0] == 0.0 and b.use(x, pos, dirv, valid, ysub(b, None))[0] < 1e-9 and b.use(x, pos, dirv, valid, ysub(b, dict(past, valid=torch.zeros_like(past["valid"]))))[0] < 1e-9
    y = ysub(b, past)
    d2_, x2_, n_ = b.use(x, pos, dirv, valid, y)
    y0 = ysub(b, None)
    m = valid[..., None].double()
    assert abs(d2_ - float((((y - y0).double() * m) ** 2).sum())) < 1e-6 * max(d2_, 1) and abs(x2_ - float(((x.double() * m) ** 2).sum())) < 1e-6 * x2_ and n_ == int(valid.sum()) and d2_ > 0
    um = UseMeter(b)                                                                                       # the hook: calls without a past are not counted, the sums are those of `use`, `take` clears
    b(x, pos, dirv, valid, None), b(x, pos, dirv, valid, {})
    assert um.take() is None
    yh = b(x, pos, dirv, valid, past)
    s3 = um.take_sums()
    assert s3 == b.use(x, pos, dirv, valid, yh - x) and um.take() is None
    b(x, pos, dirv, valid, past), b(x, pos, dirv, valid, past)
    u2, n2 = um.take()
    assert abs(u2 - math.sqrt(s3[0] / s3[1])) < 1e-9 and n2 == 2 * s3[2], "u over two calls"
    um.detach()
    b(x, pos, dirv, valid, past)
    assert um.take() is None, "a detached meter still counts"
    # a padding-only batch row and K = 512 (memory below 2 GB on the CPU, bf16 autocast finite)
    bb = _trained(3)
    big, bigp = _frame(2, 512, 4, n_invalid=100), _frame(2, 512, 5, n_invalid=50)
    xb = big["x"].clone().requires_grad_()
    ob = bb(xb, big["pos"], big["dirv"], big["valid"], _past_of(bigp, big["e"]))
    ob.square().mean().backward()
    assert ob.shape == (2, 512, D) and torch.isfinite(ob).all() and torch.isfinite(xb.grad).all()
    import resource
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 < 2048, "K = 512, batch 2, forward and backward: more than 2 GB"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(bb(big["x"], big["pos"], big["dirv"], big["valid"], _past_of(bigp, big["e"]))).all()
    print("jpast selftest ok")


if __name__ == "__main__":
    import sys
    assert len(sys.argv) > 1 and sys.argv[1] == "selftest", "usage: jpast.py selftest"
    selftest()
