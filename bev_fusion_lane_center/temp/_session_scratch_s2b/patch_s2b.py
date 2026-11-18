"""E387 (S2b) edits of the restored E368 files in a scratch tree: FusedPastBlock + its selftest in jpast.py, the mutants in test_jpast.py. usage: python patch_s2b.py <tree root> (asserted anchors; writes the files)"""
import os
import sys

root = os.path.abspath(sys.argv[1])


def edit(rel, pairs):
    p = os.path.join(root, rel)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (rel, s.count(old), old[:80])
        s = s.replace(old, new)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)


FUSED = '''class FusedPastBlock(PastBlock):
    """E387 (S2b, D121): the cross-attention of `PastBlock` followed by a learned FUSION of the cross output c and the input x, where it makes sense. Per valid token: c = o(the attended values) (the output projection at its ORDINARY initialisation: c is a live signal);
    a (h,) = per head the attention mass on REAL past keys (1 - the none mass: how much the token found in the past at all); gp (7,) = the head-mean attention weight on the real keys times the 7 cross-geometry features of the pair, summed over the keys (is the found past
    token where the token expects its predecessor?); feat = [LN_x(x), LN_c(c), a, gp] (2 d + h + 7 = 139 at d 64, 4 heads); y = W2 GELU(W1 feat) (139 -> 64 -> 64, W2 ZERO-initialised); g = sigmoid(w_g feat + b_g) (b_g = -2.0: the gate starts at 0.119 and a token passes
    unchanged where the past does not help); output x + g * y, padding queries keep x. The start is the head without the block (y = 0 exactly), the zero-init sits at the MLP output so that o and the MLP do not both start dead. A first frame / an all-invalid past: the
    softmax has the none key only, a = 0, gp = 0, c = the none value through o. 30,932 parameters at d 64 (17,416 of the attention part + 2 x 128 + 8,960 + 4,160 + 140). Same hook, same state, same kill measure (`use` through `delta`) as `PastBlock`."""

    def __init__(self, d=D, heads=4, geo_hidden=32, hidden=64, gate_bias=-2.0):
        super().__init__(d, heads, geo_hidden)
        self.o.reset_parameters()                                           # c = o(attended values) is a live signal: the zero-init moves to the MLP output
        nf = 2 * d + heads + NGEO
        self.ln_x, self.ln_c = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp1, self.mlp2 = nn.Linear(nf, hidden), nn.Linear(hidden, d)
        self.gate = nn.Linear(nf, 1)
        nn.init.zeros_(self.mlp2.weight)
        nn.init.zeros_(self.mlp2.bias)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.constant_(self.gate.bias, gate_bias)

    def parts(self, x, pos, dirv, valid, past):
        """-> (y (B, K, D) the fusion update before the gate, gate (B, K, 1), a (B, K, h) float32 the real-key mass per head, gp (B, K, NGEO) float32 the pooled cross-geometry)"""
        B, K, D_ = x.shape
        att, v, g = self.attend_g(x, pos, dirv, valid, past)
        vn = self.v_none[None, :, None, :].to(x.dtype)
        out = att[..., -1:].to(x.dtype) * vn if v is None else att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn
        c = self.o(out.transpose(1, 2).reshape(B, K, D_))
        a = (1.0 - att[..., -1]).transpose(1, 2)                                                   # (B, K, h)
        gp = torch.zeros(B, K, NGEO, dtype=a.dtype, device=a.device) if g is None else torch.einsum("bkj,bkjf->bkf", att[..., :-1].mean(1), g.float())
        feat = torch.cat([self.ln_x(x), self.ln_c(c), a.to(x.dtype), gp.to(x.dtype)], -1)
        return self.mlp2(F.gelu(self.mlp1(feat))), torch.sigmoid(self.gate(feat)), a, gp

    def delta(self, x, pos, dirv, valid, past):
        y, gate, _, _ = self.parts(x, pos, dirv, valid, past)
        return gate * y * valid.to(x.dtype)[..., None]


'''

REF = '''def _trained_fused(seed=0, **kw):
    torch.manual_seed(seed)
    b = FusedPastBlock(**kw)
    for p in (b.mlp2.weight, b.mlp2.bias, b.gate.weight):
        nn.init.normal_(p, std=0.3)
    nn.init.normal_(b.gate.bias, std=0.5)
    nn.init.normal_(b.none_b, std=0.5)
    return b


def reference_fused(b, x, pos, dirv, valid, past):
    """FusedPastBlock by loops over batch, head, query and key (plain float64 arithmetic of its docstring): the reference of the fused selftest"""
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
            att_out, mass, gp = torch.zeros(b.h, b.dh, dtype=torch.float64), torch.zeros(b.h, dtype=torch.float64), torch.zeros(NGEO, dtype=torch.float64)
            for h in range(b.h):
                zs, vs, geos = [], [], []
                if past:
                    for j in range(Kp):
                        if not bool(past["valid"][bi, j]):
                            continue
                        tq, tk = dirv[bi, i].double(), past["dirv"][bi, j].double()
                        ch = past["pos"][bi, j].double() - pos[bi, i].double()
                        nq = torch.stack([-tq[1], tq[0]])
                        geo = torch.stack([tq @ tk, tq[0] * tk[1] - tq[1] * tk[0], ch @ tq / 20, ch @ nq / 20, ch.norm() / 20, past["e_cur"][bi, i].double(), past["e"][bi, j].double()])
                        z = q[bi, i, h] @ kv[bi, j, 0, h] / math.sqrt(b.dh) + lin(F.gelu(lin(geo, "bias.0")), "bias.2")[h]
                        if past.get("w") is not None:
                            z = z + math.log(float(past["w"][bi, j]) + 1e-3)
                        zs.append(z)
                        vs.append(kv[bi, j, 1, h])
                        geos.append(geo)
                zs.append(q[bi, i, h] @ P["k_none"][h] / math.sqrt(b.dh) + P["none_b"][h])
                vs.append(P["v_none"][h])
                a_ = torch.softmax(torch.stack(zs), 0)
                att_out[h] = (a_[:, None] * torch.stack(vs)).sum(0)
                mass[h] = 1 - a_[-1]
                if geos:
                    gp = gp + (a_[:-1, None] * torch.stack(geos)).sum(0) / b.h
            c = lin(att_out.reshape(-1), "o")
            feat = torch.cat([ln(x[bi, i].double(), "ln_x"), ln(c, "ln_c"), mass, gp])
            y = lin(F.gelu(lin(feat, "mlp1")), "mlp2")
            out[bi, i] = out[bi, i] + torch.sigmoid(lin(feat, "gate")) * y
    return out


def selftest_fused():
    torch.manual_seed(0)
    B, K, Kp = 2, 7, 6
    f, fp = _frame(B, K, 1, n_invalid=2), _frame(B, Kp, 2, n_invalid=1)
    past = _past_of(fp, f["e"])
    x, pos, dirv, valid = f["x"], f["pos"], f["dirv"], f["valid"]
    # the size (a pinned number: a changed block is a new card) and the init: the MLP output is zero, the gate starts at sigmoid(-2)
    b0 = FusedPastBlock()
    assert sum(p.numel() for p in b0.parameters()) == 30932, sum(p.numel() for p in b0.parameters())
    assert sum(p.numel() for p in b0.parameters()) - sum(p.numel() for p in PastBlock().parameters()) == 13516
    assert float(b0.gate.bias) == -2.0 and abs(float(torch.sigmoid(b0.gate.bias)) - 0.1192) < 1e-4 and float(b0.mlp2.weight.abs().sum()) == 0.0 and float(b0.mlp2.bias.abs().sum()) == 0.0 and float(b0.o.weight.abs().sum()) > 0
    gt_ = torch.sigmoid(b0.gate(torch.randn(500, 139))).detach()
    assert abs(float(gt_.mean()) - 0.119) < 0.02, "the gate starts near 0.119"
    # the start is the head without the block: exactly the identity, with and without a past
    for pa in (None, {}, past):
        assert torch.equal(b0(x, pos, dirv, valid, pa), x), "the zero-initialised fusion is not the identity"
    # the loop reference of the whole block (trained-like weights; a past with a padding key; key weights; no key weights)
    b = _trained_fused()
    for pa in (None, past, dict(past, w=None)):
        got = b(x, pos, dirv, valid, pa).double()
        want = reference_fused(b, x, pos, dirv, valid, pa)
        assert (got - want).abs().max() < 1e-4, ("fused reference", None if pa is None else sorted(pa), float((got - want).abs().max()))
    assert float((b(x, pos, dirv, valid, past) - x).abs().max()) > 0.05, "the trained-like block changes nothing (a test of the test)"
    # the evidence by hand: a = 1 - the none mass per head; no past or an all-invalid past: a = 0, gp = 0 and the output is a function of x alone
    att, _ = b.attend(x, pos, dirv, valid, past)
    y, gate, a, gp = b.parts(x, pos, dirv, valid, past)
    assert a.shape == (B, K, 4) and torch.allclose(a, (1 - att[..., -1]).transpose(1, 2)) and (a > 0).all() and (a < 1).all()
    assert gp.shape == (B, K, NGEO) and gate.shape == (B, K, 1)
    wk = att[..., :-1].mean(1)
    _, _, g7 = b.attend_g(x, pos, dirv, valid, past)
    assert torch.allclose(gp, torch.einsum("bkj,bkjf->bkf", wk, g7), atol=1e-6)
    for pa in (None, {}, dict(past, valid=torch.zeros_like(past["valid"]))):
        _, _, a0, gp0 = b.parts(x, pos, dirv, valid, pa)
        assert float(a0.abs().max()) < 1e-6, "no real key: the evidence is 0"
        assert float(gp0.abs().max()) < 1e-6
    first = b(x, pos, dirv, valid, None)
    assert torch.equal(first, b(x, pos, dirv, valid, {})) and torch.allclose(first, b(x, pos, dirv, valid, dict(past, valid=torch.zeros_like(past["valid"]))), atol=1e-6), "a first frame is the all-invalid past"
    # padding keys are masked, padding queries keep x, a permutation of the past tokens leaves the output unchanged
    junk = dict(past, x=past["x"].clone(), pos=past["pos"].clone(), dirv=past["dirv"].clone(), e=past["e"].clone(), w=past["w"].clone())
    pad = ~past["valid"]
    junk["x"][pad], junk["pos"][pad], junk["e"][pad], junk["w"][pad] = 9.0, 99.0, 0.77, 5.0
    assert torch.equal(b(x, pos, dirv, valid, past), b(x, pos, dirv, valid, junk)), "a padding key changed the output"
    assert torch.equal(b(x, pos, dirv, valid, past)[~valid], x[~valid]), "a padding query was changed"
    perm = torch.randperm(Kp)
    assert torch.allclose(b(x, pos, dirv, valid, past), b(x, pos, dirv, valid, {k: v[:, perm] for k, v in past.items() if k != "e_cur"} | {"e_cur": f["e"]}), atol=1e-5), "not permutation invariant"
    sh = torch.tensor([13.0, -7.0])
    assert torch.allclose(b(x, pos, dirv, valid, past), b(x, pos + sh, dirv, valid, dict(past, pos=past["pos"] + sh)), atol=1e-4), "a common translation changed the output"
    # the start learns: at init only the MLP output has a gradient (y = 0 stops the rest); after ONE step on it every parameter has one (no dead start)
    b1 = FusedPastBlock()
    b1(x, pos, dirv, valid, past).square().sum().backward()
    live = {n for n, p in b1.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
    assert live == {"mlp2.weight", "mlp2.bias"}, live
    with torch.no_grad():
        for p in (b1.mlp2.weight, b1.mlp2.bias):
            p -= 0.1 * p.grad
    b1.zero_grad()
    xg = x.clone().requires_grad_()
    pg = dict(past, x=past["x"].clone().requires_grad_())
    b1(xg, pos, dirv, valid, pg).square().sum().backward()
    dead = [n for n, p in b1.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0]
    assert not dead and float(xg.grad.abs().sum()) > 0 and float(pg["x"].grad.abs().sum()) > 0, f"dead after one step: {dead}"
    # the kill measure through `delta`: u > 0 for a block that reads the content (trained-like), 0 at init
    ysub = b(x, pos, dirv, valid, past) - x
    su = b.use(x, pos, dirv, valid, ysub, past)
    assert su["d2_other"] > 1e-6 and su["d2_none"] > 1e-6 and su["n_other"] == int(valid.sum()), su
    s0 = b0.use(x, pos, dirv, valid, b0(x, pos, dirv, valid, past) - x, past)
    assert s0["d2_other"] == 0.0 and s0["d2_none"] == 0.0
    # K = 512 and bf16 autocast: finite (the evidence and the pooled geometry stay float32)
    bb = _trained_fused(3)
    big, bigp = _frame(2, 512, 4, n_invalid=100), _frame(2, 512, 5, n_invalid=50)
    xb = big["x"].clone().requires_grad_()
    ob = bb(xb, big["pos"], big["dirv"], big["valid"], _past_of(bigp, big["e"]))
    ob.square().mean().backward()
    assert ob.shape == (2, 512, D) and torch.isfinite(ob).all() and torch.isfinite(xb.grad).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(bb(big["x"], big["pos"], big["dirv"], big["valid"], _past_of(bigp, big["e"]))).all()
    print("jpast fused selftest ok")


'''

edit("training/grid/jpast.py", [
    ("import torch.nn as nn\n", "import torch.nn as nn\nimport torch.nn.functional as F\n"),
    ("    def attend(self, x, pos, dirv, valid, past):\n        \"\"\"-> (att (B, h, K, Kp + 1) float32:", "    def attend_g(self, x, pos, dirv, valid, past):\n        \"\"\"-> (att, v, g): `attend` and the cross-geometry g (B, K, Kp, NGEO) of every (query, key) pair (None without a past); att (B, h, K, Kp + 1) float32:"),
    ("            return z_none[..., None].softmax(-1), None\n", "            return z_none[..., None].softmax(-1), None, None\n"),
    ("        return torch.cat([z, z_none[..., None]], -1).softmax(-1), v\n\n    def delta(", "        return torch.cat([z, z_none[..., None]], -1).softmax(-1), v, g\n\n    def attend(self, x, pos, dirv, valid, past):\n        \"\"\"-> (att (B, h, K, Kp + 1) float32, v (B, h, Kp, dh) or None): see `attend_g`\"\"\"\n        att, v, _ = self.attend_g(x, pos, dirv, valid, past)\n        return att, v\n\n    def delta("),
    ("# ------------------------------------------------------------------------------------------------------------------------------ selftest", FUSED + "# ------------------------------------------------------------------------------------------------------------------------------ selftest"),
    ("def selftest():\n    torch.manual_seed(0)\n    B, K, Kp = 2, 7, 6\n    f, fp = _frame(B, K, 1, n_invalid=2), _frame(B, Kp, 2, n_invalid=1)\n    e_cur = f[\"e\"]", REF + "def selftest():\n    torch.manual_seed(0)\n    B, K, Kp = 2, 7, 6\n    f, fp = _frame(B, K, 1, n_invalid=2), _frame(B, Kp, 2, n_invalid=1)\n    e_cur = f[\"e\"]"),
    ("    print(\"jpast selftest ok\")", "    selftest_fused()\n    print(\"jpast selftest ok\")"),
])

edit("training/grid/test_jpast.py", [
    ('("the none key is missing", "return torch.cat([z, z_none[..., None]], -1).softmax(-1), v", "return z.softmax(-1), v"),', '("the none key is missing", "return torch.cat([z, z_none[..., None]], -1).softmax(-1), v, g", "return z.softmax(-1), v, g"),'),
    ('    ("u: padding tokens count in the difference", "m = valid.to(x.dtype)[..., None]\\n        sq = ", "m = torch.ones_like(x[..., :1])\\n        sq = "),\n]',
     '''    ("u: padding tokens count in the difference", "m = valid.to(x.dtype)[..., None]\\n        sq = ", "m = torch.ones_like(x[..., :1])\\n        sq = "),
    # E387 (S2b): the fusion block
    ("fused: the gate is missing", "return gate * y * valid.to(x.dtype)[..., None]", "return y * valid.to(x.dtype)[..., None]"),
    ("fused: the gate is not a sigmoid", "torch.sigmoid(self.gate(feat))", "self.gate(feat)"),
    ("fused: the padding queries are changed", "return gate * y * valid.to(x.dtype)[..., None]", "return gate * y"),
    ("fused: the evidence is the none mass", "a = (1.0 - att[..., -1]).transpose(1, 2)", "a = att[..., -1].transpose(1, 2)"),
    ("fused: the pooled geometry is not weighted by the attention", 'torch.einsum("bkj,bkjf->bkf", att[..., :-1].mean(1), g.float())', 'torch.einsum("bkj,bkjf->bkf", (att[..., :-1].mean(1) > 0).float(), g.float())'),
    ("fused: the MLP output is not zero-initialised", "nn.init.zeros_(self.mlp2.weight)", "pass"),
    ("fused: the gate bias is not initialised", "nn.init.constant_(self.gate.bias, gate_bias)", "pass"),
    ("fused: the gate starts open", "gate_bias=-2.0", "gate_bias=2.0"),
    ("fused: the two normalised inputs are swapped", "torch.cat([self.ln_x(x), self.ln_c(c),", "torch.cat([self.ln_c(c), self.ln_x(x),"),
    ("fused: the input is not normalised", "torch.cat([self.ln_x(x), self.ln_c(c),", "torch.cat([x, self.ln_c(c),"),
]'''),
])
print("patched", root)
