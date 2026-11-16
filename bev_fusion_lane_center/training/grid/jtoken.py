"""E365 (stage S1 of the token champion, D113): the E2E token head on the PREDICTED nodes of the frozen lane head B'; the wiring is the per-source successor softmax with a none slot.

    head = JTokenHead()                                       # 0.127 M parameters; the buffer `step` counts the training forwards (the switch of the key weights, below)
    out = head(batch)                                         # batch of `data/lane_ptok_data.loader` (section 2a of the card); only the INPUT keys are read, never a key that carries GT
    loss, info = jtoken_loss(out, batch)                      # the exact likelihood below + type CE + scale L1 (reads y, ylong, yscale, ytype)
    pr = probs(out); links = decode(out, thr=0.3)             # decode of record; decode_argmax(out) = the alternative (one link per source and class)
    batch = synthetic_batch(B, K, seed)                       # a random batch that follows the batch dict (tests and the trainer's --smoke)

Input keys (B frames, K tokens): tok (B,K,64), pos (B,K,2) m, dirv (B,K,2) unit, e / e_raw (B,K) logits, flags (B,K,8) logits, s / uout / uin (B,K) logits, valid (B,K) bool. All of them are constants (the cached B' is
frozen: the forward cuts the graph at them, no gradient reaches B' and none goes through the hard top-K).
Embedding: x0 = LN(Linear(tok) + MLP[Fourier(pos), dirv, e, e_raw, flags, s, uout, uin]). 3 pre-norm self-attention layers (4 heads, FFN 128): the logit of head h = q.k / 4 + bias_h(g_ij) + log(w_j + 1e-3), bias = an MLP
13 -> 32 -> H of the 13 `jtok.pair_geometry` features (the existence enters them as sigmoid(e)), padding tokens masked as keys.
Soft pruning: the key weight w_j is sigmoid(e_j) (input) in layer 1 and for the first SWITCH training steps in every layer; afterwards w_j = 1 - p_none(j) of the head's OWN type softmax (7 types, type 0 = none), read by
the SAME Linear (after a shared LayerNorm) after layer 1 (keys of layers 2 and 3) and after layer 3 (keys of the wiring layer); both readings get the type CE, and the key-bias path carries gradient into them.
Wiring layer (on top of layer 3, attention weights only): class c in {short: chord <= 8 m, long: chord > 8 m}, z^c_ij = q^c_i.k^c_j / sqrt(32) + bias^c(g_ij) + log(w_j + 1e-3) + b^c_j on the allowed gate of `jtok.pair_geometry`
(padding masked as key and source) + ONE learned none slot per class (key k^c_none and a bias): p^c(.|i) = softmax over {allowed keys} + {none}. Two handle scales per pair from an MLP of (x_i, x_j, g_ij).
Likelihood (`jtoken_loss`): per class c and source i (every valid token with at least one candidate of the class): G(i) = the in-gate tokens with y[i, j] = 1 of the class (ylong), nll_i = -log sum_{j in G(i)} p^c(j|i)
when G(i) is non-empty, -log p^c(none|i) when it is empty (also when its GT links lie outside the gate: counted in info); per frame the mean over the sources of the class, then over the frames. + 0.5 x (type CE after layer 1
+ type CE after layer 3, per frame the mean over the valid tokens) + 0.5 x L1 on the handle scales of the long GT links. Tests: test_jtoken.py
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "..", "../data", "../data/prep"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jtok as JT  # noqa: E402
import jtok2 as J2  # noqa: E402

SHORT, NEG, NG = JT.SHORT, J2.NEG, JT.NG
TOK, D, NTYPE = 64, 64, 7                                                   # B's node token, the width, the node types (0 none, 1 lane point, 2 fork, 3 merge, 4 entry, 5 exit, 6 boundary)
SWITCH = 1000                                                               # training steps with the key weight sigmoid(e) in every layer; then the head's own 1 - p_none
BANDS = 4
NIN = 2 + 4 * BANDS + 2 + 1 + 1 + 8 + 3                                     # pos / HALF, its Fourier features, dirv, e, e_raw, 8 flags, s, uout, uin
INPUTS = ("tok", "pos", "dirv", "e", "e_raw", "flags", "s", "uout", "uin", "valid")      # the only batch keys the forward reads
GT_KEYS = ("y", "ylong", "yscale", "ytype")                                 # the keys of the batch that carry GT (the loss reads them, the forward never)


def _const(t):
    """a cached input: no gradient into B' (the one place that cuts the graph)"""
    return t.detach()


def fourier(pos):
    """(B, K, 2) m -> (B, K, 4 * BANDS) sin / cos of pos / HALF at BANDS octaves"""
    x = pos[..., None] / JT.HALF * (math.pi * 2.0 ** torch.arange(BANDS, device=pos.device, dtype=pos.dtype))
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2)


def pair_mlp(n_out, hid=32):
    return nn.Sequential(nn.Linear(NG, hid), nn.GELU(), nn.Linear(hid, n_out))


class SelfLayer(nn.Module):
    """a pre-norm self-attention layer with the relative-geometry bias and the key weight (module docstring)"""

    def __init__(self, d=D, heads=4, ffn=128):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.bias = pair_mlp(heads)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))

    def forward(self, x, g, logw, valid):
        B, K, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, K, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh) + self.bias(g).float().permute(0, 3, 1, 2) + logw[:, None, None, :]
        a = a.masked_fill(~valid[:, None, None, :], NEG).softmax(-1)
        x = x + self.o((a.to(v.dtype) @ v).transpose(1, 2).reshape(B, K, -1))
        return x + self.ffn(self.ln2(x))


def wiring_logp(z, zn, cand):
    """z (B, 2, K, K) key logits, zn (B, 2, K) none logits, cand (B, 2, K, K) bool -> (B, 2, K, K + 1) float32 log-probabilities of the per-source softmax over {the candidates} + {none} (the LAST column); a key outside
    `cand` has probability exactly 0 and a source without a candidate has p(none) = 1"""
    return torch.cat([z.float().masked_fill(~cand, NEG), zn.float()[..., None]], -1).log_softmax(-1)


class JTokenHead(nn.Module):
    def __init__(self, d=D, heads=4, layers=3, ffn=128, dq=32):
        super().__init__()
        self.dq = dq
        self.proj = nn.Linear(TOK, d)
        self.mlp = nn.Sequential(nn.Linear(NIN, d), nn.GELU(), nn.Linear(d, d))
        self.ln0 = nn.LayerNorm(d)
        self.layers = nn.ModuleList([SelfLayer(d, heads, ffn) for _ in range(layers)])
        self.tnorm, self.types = nn.LayerNorm(d), nn.Linear(d, NTYPE)       # ONE reading for both positions (after layer 1 and after the last layer)
        self.lnw = nn.LayerNorm(d)
        self.wq, self.wk = nn.Linear(d, 2 * dq), nn.Linear(d, 2 * dq)       # the two classes (short, long)
        self.wbias, self.dest = pair_mlp(2), nn.Linear(d, 2)                # bias^c(g_ij), the destination evidence b^c_j
        self.none_k, self.none_b = nn.Parameter(torch.randn(2, dq) * 0.1), nn.Parameter(torch.zeros(2))
        self.sc_i, self.sc_j, self.sc_g, self.sc_o = nn.Linear(d, 32), nn.Linear(d, 32, bias=False), nn.Linear(NG, 32, bias=False), nn.Linear(32, 2)      # the first Linear of (x_i, x_j, g_ij) split into three
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def read_types(self, x):
        t = self.types(self.tnorm(x)).float()
        return t, 1.0 - t.softmax(-1)[..., 0]                               # logits, 1 - p_none

    def forward(self, batch):
        b = {k: _const(batch[k]) for k in INPUTS}
        valid, pos, dirv = b["valid"], b["pos"].float(), b["dirv"].float()
        w_in = b["e"].float().sigmoid()
        g, allowed = JT.pair_geometry(pos, dirv, valid, e=w_in)
        lw = lambda w: (w + 1e-3).log()  # noqa: E731
        learned = bool(self.step >= SWITCH)
        feat = torch.cat([pos / JT.HALF, fourier(pos), dirv, b["e"][..., None], b["e_raw"][..., None], b["flags"], b["s"][..., None], b["uout"][..., None], b["uin"][..., None]], -1).float()
        x = self.ln0(self.proj(b["tok"].float()) + self.mlp(feat))
        x = self.layers[0](x, g, lw(w_in), valid)
        types1, w1 = self.read_types(x)
        for layer in self.layers[1:]:
            x = layer(x, g, lw(w1 if learned else w_in), valid)
        types3, w3 = self.read_types(x)
        # the wiring layer
        B, K, _ = x.shape
        xn = self.lnw(x)
        q, k = (m(xn).view(B, K, 2, self.dq) for m in (self.wq, self.wk))
        z = torch.einsum("bicd,bjcd->bcij", q, k).float() / math.sqrt(self.dq) + self.wbias(g).float().permute(0, 3, 1, 2) + lw(w3 if learned else w_in)[:, None, None, :] + self.dest(xn).float().permute(0, 2, 1)[:, :, None, :]
        zn = torch.einsum("bicd,cd->bci", q, self.none_k).float() / math.sqrt(self.dq) + self.none_b[None, :, None]
        chord = g[..., 2] * 20.0
        cand = torch.stack([allowed & (chord <= SHORT), allowed & (chord > SHORT)], 1)
        scale = 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[:, :, None] + self.sc_j(xn)[:, None] + self.sc_g(g))).float().sigmoid()
        if self.training:
            self.step += 1
        return dict(logp=wiring_logp(z, zn, cand), cand=cand, scale=scale, types=types3, types1=types1, valid=valid, chord=chord)


# ------------------------------------------------------------------------------------------------------------------------------ probabilities and decodes
def probs(out):
    """-> p (B, K, K) the probability of the link i -> j in the class of the pair (0 outside the gate), pnone (B, K, K) the none probability of the source i in the class of the pair, pt (B, K, 7) the type probabilities,
    keep (B, K) the kept nodes (valid and p_none < 0.5, the output rule), cand (B, K, K) the allowed gate, longm (B, K, K) chord above 8 m"""
    K = out["cand"].shape[-1]
    cand = out["cand"]
    p = (out["logp"][..., :K].exp() * cand).sum(1)
    pnone = (out["logp"][..., K].exp()[..., None] * cand).sum(1)
    pt = out["types"].float().softmax(-1)
    return dict(p=p, pnone=pnone, pt=pt, keep=out["valid"] & (pt[..., 0] < 0.5), cand=cand.any(1), longm=out["chord"] > SHORT)


def _links(pr, link):
    link = link & pr["cand"] & pr["keep"][:, :, None] & pr["keep"][:, None, :]
    return dict(keep=pr["keep"], link=link, long=link & pr["longm"], p=pr["p"])


def decode(out, thr=0.3):
    """the decode of record (output only): link i -> j when p^c(j|i) >= thr and p^c(j|i) >= p^c(none|i) and both ends are kept nodes (a fork source keeps both branches) -> keep (B, K), link / long (B, K, K) bool, p"""
    pr = probs(out)
    return _links(pr, (pr["p"] >= thr) & (pr["p"] >= pr["pnone"]))


def decode_argmax(out):
    """the alternative: per source and class the argmax over {none} + the candidates, a link when it is a candidate (one link per source and class), both ends kept"""
    K = out["cand"].shape[-1]
    am = out["logp"].argmax(-1)
    hit = F.one_hot(am.clamp(max=K - 1), K).bool() & (am < K)[..., None]
    return _links(probs(out), hit.any(1))


# ------------------------------------------------------------------------------------------------------------------------------ loss
def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5):
    """the likelihood of the module docstring -> (loss, info); batch: y (B, K, K), ylong (B, K, K), yscale (B, K, K, 2), ytype (B, K). info: nll_s, nll_l, ce1, ce3, scale_l1 and the counts gt_links, pos_pairs (in-gate GT
    links), gt_outside (links with no gate), gt_class (in the gate, but the chord class of the tokens is not the class of the GT link)"""
    dev, cand, lp, valid = out["logp"].device, out["cand"], out["logp"], out["valid"]
    K = cand.shape[-1]
    y, yl = batch["y"].to(dev) > 0, batch["ylong"].to(dev).bool()
    G = torch.stack([y & ~yl, y & yl], 1) & cand                           # (B, 2, K, K): the in-gate GT successors of the class
    nll = torch.where(G.any(-1), -torch.logsumexp(lp[..., :K].masked_fill(~G, NEG), -1), -lp[..., K])
    rows = cand.any(-1) & valid[:, None]                                    # the sources of a class: valid tokens with a candidate of it
    nll_s, nll_l = (J2._frame_mean(nll[:, c], rows[:, c]) for c in range(2))
    yt = batch["ytype"].to(dev).long()
    ce1, ce3 = (J2._frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt), valid) for t in (out["types1"], out["types"]))
    loss = nll_s + nll_l + type_w * (ce1 + ce3)
    info = dict(nll_s=float(nll_s), nll_l=float(nll_l), ce1=float(ce1), ce3=float(ce3), gt_links=float(y.sum()), pos_pairs=float(G.sum()), gt_outside=float((y & ~cand.any(1)).sum()),
                gt_class=float((y & cand.any(1) & (yl != (out["chord"] > SHORT))).sum()))
    ys = batch["yscale"].to(dev).float()
    m = G[:, 1] & torch.isfinite(ys[..., 0])
    if m.any():
        l1 = (out["scale"].float()[m] - ys[m]).abs().mean()
        loss = loss + scale_w * l1
        info["scale_l1"] = float(l1)
    return loss, info


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------------------------------------------------------------------ a synthetic batch
def synthetic_batch(B=8, K=48, seed=0):
    """random frames that follow the batch dict of the card (2a): per frame a lane of 10 tokens 2 m apart, a second lane 12 m behind its end (one long link), a third lane elsewhere, 4 duplicate tokens 0.5 m beside
    a lane token (they are in G(i) of its predecessor, assigned to no GT node), padding to K. Types and flags follow the token role, so a head can fit them."""
    rng = np.random.RandomState(seed)
    f = lambda *s: np.zeros(s, np.float32)  # noqa: E731
    pos, dirv, valid, ytype, y, ylong = f(B, K, 2), f(B, K, 2), np.zeros((B, K), bool), np.zeros((B, K), np.int64), np.zeros((B, K, K), np.uint8), np.zeros((B, K, K), bool)
    yscale = np.full((B, K, K, 2), np.nan, np.float32)
    dirv[..., 1] = 1.0
    for b in range(B):
        th = rng.uniform(0, 2 * np.pi, 3)
        u = lambda t: np.array([-np.sin(t), np.cos(t)])  # noqa: E731
        start = [rng.uniform(-25, 25, 2) for _ in range(3)]
        start[1] = start[0] + u(th[0]) * (2 * 9 + 12)
        th[1] = th[0]
        lanes = [list(range(10 * l, 10 * l + 10)) for l in range(3)]
        for l, ids in enumerate(lanes):
            for n, i in enumerate(ids):
                pos[b, i], dirv[b, i], valid[b, i], ytype[b, i] = start[l] + u(th[l]) * 2.0 * n, u(th[l]), True, 2 if n == 0 and l == 1 else 1
                if n:
                    y[b, i - 1, i] = 1
        y[b, 9, 10], ylong[b, 9, 10], yscale[b, 9, 10] = 1, True, (0.3, 0.4)
        for n, i in enumerate(range(30, 34)):                              # duplicates of tokens 3, 12, 22, 25
            src = (3, 12, 22, 25)[n]
            pos[b, i], dirv[b, i], valid[b, i] = pos[b, src] + u(th[src // 10] + np.pi / 2) * 0.5, dirv[b, src], True
            y[b, src - 1, i] = 1
        for n in range(34, K - rng.randint(2, 8)):                         # extra tokens that belong to no lane
            pos[b, n], dirv[b, n], valid[b, n] = rng.uniform(-40, 40, 2), u(rng.uniform(0, 2 * np.pi)), True
    ylong |= (y > 0) & (np.linalg.norm(pos[:, None] - pos[:, :, None], axis=-1) > SHORT)
    e = np.where(ytype > 0, 2.5, np.where(valid, -1.0, -6.0)).astype(np.float32)
    flags = (-3.0 + 6.0 * (np.arange(8) == (ytype[..., None] % 8))).astype(np.float32)
    r = lambda *s: rng.randn(*s).astype(np.float32)  # noqa: E731
    return dict(tok=torch.from_numpy(r(B, K, TOK) * valid[..., None]), pos=torch.from_numpy(pos), dirv=torch.from_numpy(dirv), e=torch.from_numpy(e), e_raw=torch.from_numpy(e + 0.3 * r(B, K)),
                flags=torch.from_numpy(flags), s=torch.from_numpy(r(B, K)), uout=torch.from_numpy(r(B, K)), uin=torch.from_numpy(r(B, K)), valid=torch.from_numpy(valid), ytype=torch.from_numpy(ytype),
                y=torch.from_numpy(y), ylong=torch.from_numpy(ylong), yscale=torch.from_numpy(yscale))
