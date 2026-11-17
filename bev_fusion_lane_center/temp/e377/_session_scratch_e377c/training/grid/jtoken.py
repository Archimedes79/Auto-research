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
Likelihood (`jtoken_loss`): per class c and source i (every valid token with at least one candidate of the class): G(i) = the in-gate tokens with y[i, j] = 1 whose pair lies in the class by the TOKEN chord (as `jtok2.gt_classes`; ylong is only a statistic and the long scale target), nll_i = -log sum_{j in G(i)} p^c(j|i)
(E376 `pos_loss="max"`: -log max_{j in G(i)} p^c(j|i)) when G(i) is non-empty, -log p^c(none|i) when it is empty (also when its GT links lie outside the gate: counted in info); per frame the mean over the sources of the class, then over the frames. + 0.5 x (type CE after layer 1
+ type CE after layer 3, per frame the mean over the valid tokens) + 0.5 x L1 on the handle scales of the long GT links. E377 `none_weight_near`: the none tokens near an assigned token (`near_none`, a label of the batch)
get their own none weight in both places the none weight acts. Tests: test_jtoken.py
c-rule (`rule_links`, THE one code path of the rule: the arm `rule` of `jtoken_eval.py ctl`, the E369 prior and the E370 teacher call it): E357's `lo_near` on the valid tokens with B' e_p >= 0.5.
E369 (`JTokenHead(prior="rule")`): c-rule's links r and B''s existence as a FIXED prior: z^c_ij += lam r^c_ij (r^c = r inside the gate of class c), the none slot of class c += lam (1 - has^c_i) + mu, the type none logit
+= log 6 - e_j inside `read_types` (both readings: at zero type logits p_none = sigmoid(-e), and the key weight 1 - p_none of the switch carries it); lam, mu are buffers; the type Linear, `wk`, the last layer of `wbias` and
`dest` start at zero and the handle scales at c-rule's 1/3, so the step-0 head delivers c-rule (inside the gate). E370 (`rule_teacher`): c-rule's links and B''s own types as the TARGETS of `jtoken_loss` (stage A; no GT key read).
S2 hook: `JTokenHead(past_block=m)`, `forward(batch, past)`: after self-attention layer 1, x = m(x, pos, dirv, valid, past); `past` is passed UNTOUCHED (this file reads none of its keys); out['x'] (B, K, D) = the final
token states, out['w'] (B, K) = 1 - p_none of the last reading.
S3 hook: `JTokenHead(map_block=m)`, `forward(batch, past, maps)`: after self-attention layer 2, x = m(x, pos, dirv, valid, maps); `maps` is passed UNTOUCHED (this file reads none of its keys; `{}` = no map: the block
itself must then act as the identity, the head does not special-case it). Order with both hooks: layer 1 -> past -> layer 2 -> map -> layer 3. No block, or past / maps None: the plain head bit for bit.
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

SHORT, NG = JT.SHORT, JT.NG
NEG = -1e4                                                                  # the fill of a masked logit (as `jtok2.NEG`)
TOK, D, NTYPE = 64, 64, 7                                                   # B's node token, the width, the node types (0 none, 1 lane point, 2 fork, 3 merge, 4 entry, 5 exit, 6 boundary)
SWITCH = 1000                                                               # training steps with the key weight sigmoid(e) in every layer; then the head's own 1 - p_none
BANDS = 4
NIN = 2 + 4 * BANDS + 2 + 1 + 1 + 8 + 3                                     # pos / HALF, its Fourier features, dirv, e, e_raw, 8 flags, s, uout, uin
INPUTS = ("tok", "pos", "dirv", "e", "e_raw", "flags", "s", "uout", "uin", "valid")      # the only batch keys the forward reads
GT_KEYS = ("y", "ylong", "yscale", "ytype")                                 # the keys of the batch that carry GT (the loss reads them, the forward never)
RULE_P = 0.5                                                                # c-rule's nodes: the valid tokens with B' e_p >= 0.5
LAM, MU = 12.0, 10.4                                                        # E369 prior: a rule link starts at p 0.73 - 0.82 (5 - 95 % of the destination key bias, TRAIN sample seed 0), the none slot at lam + mu without one
POS_LOSS = ("sum", "max")                                                   # the likelihood of a source with GT successors (`_source_nll`): -log sum over its group (default) or -log max (E376)
NEAR_R, NEAR_COS = 2.0, 0.5                                                 # E377 `near_none`: within NEAR_R m (inclusive) of an assigned token whose heading has a cosine >= NEAR_COS (2 m covers 98.6 % of the TRAIN duplicates)


def _const(t):
    """a cached input: no gradient into B' (the one place that cuts the graph)"""
    return t.detach()


def _frame_mean(x, m, w=None):
    """per frame the mean of x (B, N) over the entries of m, then the mean over the frames that have one (0 when none); w (B, N): the weight of every entry, the mean is then sum(w x) / sum(w) as torch's class-weighted cross-entropy
    (x is the already weighted term); w None = the plain mean"""
    per = (x * m).sum(1) / (m.sum(1).clamp_min(1) if w is None else (w * m).sum(1).clamp_min(1e-12))
    has = m.any(1)
    return per[has].mean() if has.any() else x.sum() * 0.0


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


# ------------------------------------------------------------------------------------------------------------------------------ c-rule (E369 prior, E370 teacher)
def _jtok_eval():
    """E357's rule code (`evaluation/jtok_eval.py`), imported on first use: only the rule options need it"""
    ev = os.path.normpath(os.path.join(HERE, "..", "..", "evaluation"))
    if ev not in sys.path:
        sys.path.append(ev)
    import jtok_eval
    return jtok_eval


def rule_links(pos, dirv, keep):
    """c-rule on one frame: pos, dirv (K, 2), keep (K,) bool (its nodes) -> (src, dst) token ids: the `lo` link of every kept token (the nearest kept token ahead within 45 degrees of its heading, <= 8 m, heading
    difference <= 60 degrees) and `lo_near`'s long link of every kept token without a `lo` successor (to the nearest kept token without a `lo` predecessor inside the allowed gate, chord > 8 m, <= 40 m)"""
    JV = _jtok_eval()
    idx = np.flatnonzero(keep)
    p, d = np.asarray(pos, float)[idx], np.asarray(dirv, float)[idx]
    src, dst = JV.rule_links(p, d)
    s2, d2 = JV.open_links(p, d, src, dst, every=False)
    return idx[np.concatenate([src, s2]).astype(int)], idx[np.concatenate([dst, d2]).astype(int)]


def rule_matrix(pos, dirv, keep):
    """c-rule's links of a batch: pos, dirv (B, K, 2), keep (B, K) bool -> (B, K, K) bool on the CPU, [b, i, j] = the link i -> j (a constant of the inputs: no gradient)"""
    pos, dirv, keep = pos.detach().cpu().double().numpy(), dirv.detach().cpu().double().numpy(), keep.detach().cpu().numpy()
    r = np.zeros(keep.shape + keep.shape[-1:], bool)
    for b in range(len(r)):
        s, d = rule_links(pos[b], dirv[b], keep[b])
        r[b, s, d] = True
    return torch.from_numpy(r)


def rule_teacher(out, batch):
    """E370 stage A: the targets of `jtoken_loss` made by c-rule and by B' itself; reads no GT key: per frame `tokens_np` (B''s e_p, flags_p, pos, dirv: the fields of c-rule), `valid` (the tokens the head saw) and the head's gate
    (out: cand, chord) -> dict(y, ylong, ytype, yskip, rule_links, rule_out_gate). The nodes are the valid tokens with e_p >= 0.5; y = c-rule's links inside the gate, ylong = y and chord > 8 m; ytype = 0 for the other tokens, else
    B''s own flags (`lane_tokcache.types_of(flags_p >= 0.5)`: split > merge > stop > exit > boundary > lane point); yskip (B, 2, K) = the sources whose rule link of the class (by chord) lies outside the gate: left out of the
    wiring loss, not taught 'none'. No yscale: stage A has no scale term (`jtoken_loss(scale_w=0)`)"""
    import lane_tokcache as LC
    cand, chord = out["cand"], out["chord"]
    valid = batch["valid"].bool().cpu().numpy()
    K = cand.shape[-1]
    r, ytype = np.zeros((len(valid), K, K), bool), np.zeros((len(valid), K), np.int64)
    for b, t in enumerate(batch["tokens_np"]):
        keep = valid[b] & (np.asarray(t["e_p"]) >= RULE_P)
        s, d = rule_links(t["pos"], t["dirv"], keep)
        r[b, s, d] = True
        ytype[b, keep] = LC.types_of(np.asarray(t["flags_p"])[keep] >= RULE_P)
    r = torch.from_numpy(r).to(cand.device)
    gate, longm = cand.any(1), chord > SHORT
    y, out_gate = r & gate, r & ~gate
    return dict(y=y.to(torch.uint8), ylong=y & longm, ytype=torch.from_numpy(ytype).to(cand.device), yskip=torch.stack([(out_gate & ~longm).any(-1), (out_gate & longm).any(-1)], 1),
                rule_links=float(r.sum()), rule_out_gate=float(out_gate.sum()))


class JTokenHead(nn.Module):
    """prior: None or "rule" (E369, module docstring; lam / mu are buffers); past_block: None or a module called after layer 1 as past_block(x, pos, dirv, valid, past) when `forward` gets a `past` (the S2 hook);
    map_block: None or a module called after layer 2 as map_block(x, pos, dirv, valid, maps) when `forward` gets `maps` (the S3 hook)"""

    def __init__(self, d=D, heads=4, layers=3, ffn=128, dq=32, prior=None, lam=LAM, mu=MU, past_block=None, map_block=None):
        super().__init__()
        assert prior in (None, "rule"), prior
        assert map_block is None or layers >= 2, "the map hook sits after layer 2"
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
        self._step_py = 0                                                   # the Python mirror of `step` (no device sync per forward); kept by forward, set_step and load_state_dict
        self.prior = prior
        if prior:                                                           # E369: the fixed prior; the outputs of the learned part start at zero (wq, none_k keep their init: q . k and the none slot have a gradient)
            self.register_buffer("lam", torch.tensor(float(lam)))
            self.register_buffer("mu", torch.tensor(float(mu)))
            for m in (self.types, self.wk, self.wbias[-1], self.dest):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
            nn.init.zeros_(self.sc_o.weight)                                # the handle scales start at c-rule's 1/3 (other scales change topo F1: the E369 fixture)
            nn.init.constant_(self.sc_o.bias, math.log((1 / 3 - 0.05) / (1.25 - 1 / 3)))
        self.past_block = past_block
        self.map_block = map_block

    def set_step(self, n):
        self.step.fill_(n)
        self._step_py = int(n)

    def _load_from_state_dict(self, *a, **k):
        super()._load_from_state_dict(*a, **k)
        self._step_py = int(self.step)

    def read_types(self, x, none_prior=None):
        """-> (type logits, 1 - p_none); none_prior (B, K) is added to the none logit HERE, so that the key weight 1 - p_none carries it"""
        t = self.types(self.tnorm(x)).float()
        if none_prior is not None:
            t = torch.cat([t[..., :1] + none_prior[..., None], t[..., 1:]], -1)
        return t, 1.0 - t.softmax(-1)[..., 0]                               # logits, 1 - p_none

    def forward(self, batch, past=None, maps=None):
        b = {k: _const(batch[k]) for k in INPUTS}
        valid, pos, dirv = b["valid"].bool(), b["pos"].float(), b["dirv"].float()
        w_in = b["e"].float().sigmoid()
        g, allowed = JT.pair_geometry(pos, dirv, valid, e=w_in)
        lw = lambda w: (w + 1e-3).log()  # noqa: E731
        learned = self._step_py >= SWITCH
        tp = math.log(NTYPE - 1) - b["e"].float() if self.prior else None    # E369: p_none = sigmoid(-e) at zero type logits
        feat = torch.cat([pos / JT.HALF, fourier(pos), dirv, b["e"][..., None], b["e_raw"][..., None], b["flags"], b["s"][..., None], b["uout"][..., None], b["uin"][..., None]], -1).float()
        x = self.ln0(self.proj(b["tok"].float()) + self.mlp(feat))
        x = self.layers[0](x, g, lw(w_in), valid)
        if self.past_block is not None and past is not None:
            x = self.past_block(x, pos, dirv, valid, past)
        types1, w1 = self.read_types(x, tp)
        for n, layer in enumerate(self.layers[1:], 2):
            x = layer(x, g, lw(w1 if learned else w_in), valid)
            if n == 2 and self.map_block is not None and maps is not None:
                x = self.map_block(x, pos, dirv, valid, maps)
        types3, w3 = self.read_types(x, tp)
        # the wiring layer
        B, K, _ = x.shape
        xn = self.lnw(x)
        q, k = (m(xn).view(B, K, 2, self.dq) for m in (self.wq, self.wk))
        z = torch.einsum("bicd,bjcd->bcij", q, k).float() / math.sqrt(self.dq) + self.wbias(g).float().permute(0, 3, 1, 2) + lw(w3 if learned else w_in)[:, None, None, :] + self.dest(xn).float().permute(0, 2, 1)[:, :, None, :]
        zn = torch.einsum("bicd,cd->bci", q, self.none_k).float() / math.sqrt(self.dq) + self.none_b[None, :, None]
        chord = g[..., 2] * 20.0
        cand = torch.stack([allowed & (chord <= SHORT), allowed & (chord > SHORT)], 1)
        if self.prior:
            r = rule_matrix(pos, dirv, valid & (w_in >= RULE_P)).to(z.device)[:, None] & cand                 # (B, 2, K, K): c-rule's links in the class of the pair, inside the gate
            z = z + self.lam * r
            zn = zn + self.lam * ~r.any(-1) + self.mu
        scale = 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[:, :, None] + self.sc_j(xn)[:, None] + self.sc_g(g))).float().sigmoid()
        if self.training:
            self.step += 1
            self._step_py += 1
        return dict(logp=wiring_logp(z, zn, cand), cand=cand, scale=scale, types=types3, types1=types1, valid=valid, chord=chord, x=x, w=w3)


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
def _source_nll(lp, G, none_weight, pos_loss="sum"):
    """per source and class: lp (B, 2, K, K + 1) the log-probabilities of the per-source softmax (the none slot last), G (B, 2, K, K) the in-class GT successor groups -> (B, 2, K). G(i) non-empty: the likelihood of
    its GT successors, -log sum_{G(i)} p ('sum') or -log max_{G(i)} p ('max', E376: the mass on ONE member of the group; with one member the two are the same); G(i) empty: `none_weight` x -log p(none|i) (the sources
    with a GT successor are not touched by the weight)"""
    assert pos_loss in POS_LOSS, pos_loss
    K = G.shape[-1]
    g = lp[..., :K].masked_fill(~G, NEG)
    pos = -(torch.logsumexp(g, -1) if pos_loss == "sum" else g.amax(-1))
    return torch.where(G.any(-1), pos, -lp[..., K] * none_weight)


def near_none(pos, dirv, valid, ytype):
    """E377, a label of the loss from the training labels and the token geometry (nothing at inference reads it): pos, dirv (B, K, 2), valid (B, K) bool, ytype (B, K) -> (B, K) bool, the valid none tokens (ytype 0)
    within NEAR_R m (inclusive) of a valid ASSIGNED token (ytype != 0) whose heading has a cosine >= NEAR_COS with their own: the duplicates and near-twins of a real node"""
    asg = valid & (ytype != 0)
    d2 = (pos[:, :, None] - pos[:, None]).square().sum(-1)                  # (B, K, K) squared distances: exact at NEAR_R (no sqrt, no matmul)
    cos = (dirv[:, :, None] * dirv[:, None]).sum(-1)
    return valid & (ytype == 0) & ((d2 <= NEAR_R ** 2) & (cos >= NEAR_COS) & asg[:, None, :]).any(-1)


def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5, none_weight=1.0, pos_loss="sum", none_weight_near=None):
    """the likelihood of the module docstring -> (loss, info); batch: y (B, K, K), ylong (B, K, K), yscale (B, K, K, 2), ytype (B, K). info: nll_s, nll_l, ce1, ce3, scale_l1 and the counts gt_links, pos_pairs (in-gate GT
    links), gt_outside (links with no gate), gt_class (in the gate, but `ylong` disagrees with the token chord: a statistic, the class of a link is the token chord's). The long scale target is the fitted yscale
    of the long-class links (finite where the cache fitted one). none_weight W (default 1: the loss of before): multiplies (i) the loss of the sources whose G(i) is empty (their target is the none slot), in both classes, and (ii) the type
    cross-entropy of the tokens whose type is none (class 0 weight W, the six others 1, normalised by the sum of the weights of the valid tokens of the frame, as torch's weighted mean). pos_loss 'sum' (default: the loss of
    before) or 'max' (E376): the likelihood of the sources with a non-empty G(i), in both classes (`_source_nll`); the empty-G terms, the type cross-entropy and the scale term are the same under both.
    none_weight_near W' (E377; None = W): replaces W for the near none tokens (`near_none` of the batch's pos / dirv / ytype and out's valid) in both places, their empty-G source terms (both classes) and their type
    cross-entropy weight (still normalised by the sum of the weights of the valid tokens of the frame); given, info has n_near = the near tokens per frame (the mean over the frames of the batch).
    A target batch of `rule_teacher` holds yskip (B, 2, K): those sources are left out of their class. scale_w = 0: no scale term, `yscale` is not read"""
    dev, cand, lp, valid = out["logp"].device, out["cand"], out["logp"], out["valid"]
    y, yl = batch["y"].to(dev) > 0, batch["ylong"].to(dev).bool()
    G = y[:, None] & cand                                                  # (B, 2, K, K): the in-gate GT successors of each class (the class of a pair = its token chord, as `jtok2.gt_classes`)
    yt = batch["ytype"].to(dev).long()
    wn = none_weight                                                        # the none weight of a token: of its empty-G source terms and, for a none token, of its type CE; a scalar when it is the same for all
    if none_weight_near is not None:
        near = near_none(batch["pos"].to(dev).float(), batch["dirv"].to(dev).float(), valid, yt)
        if none_weight_near != none_weight:
            wn = torch.where(near, none_weight_near, none_weight)          # (B, K)
    nll = _source_nll(lp, G, wn[:, None] if torch.is_tensor(wn) else wn, pos_loss)
    rows = cand.any(-1) & valid[:, None]                                    # the sources of a class: valid tokens with a candidate of it
    if "yskip" in batch:
        rows = rows & ~batch["yskip"].to(dev)
    nll_s, nll_l = (_frame_mean(nll[:, c], rows[:, c]) for c in range(2))
    wt = torch.where(yt == 0, wn, 1.0) if torch.is_tensor(wn) or none_weight != 1.0 else None
    ce1, ce3 = (_frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt) * (1.0 if wt is None else wt), valid, wt) for t in (out["types1"], out["types"]))
    loss = nll_s + nll_l + type_w * (ce1 + ce3)
    info = dict(nll_s=float(nll_s), nll_l=float(nll_l), ce1=float(ce1), ce3=float(ce3), gt_links=float(y.sum()), pos_pairs=float(G.sum()), gt_outside=float((y & ~cand.any(1)).sum()),
                gt_class=float((y & cand.any(1) & (yl != (out["chord"] > SHORT))).sum()))
    if none_weight_near is not None:
        info["n_near"] = float(near.sum(1).float().mean())
    if not scale_w:
        return loss, info
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
    a lane token (they are in G(i) of its predecessor, assigned to no GT node), padding to K. Types and flags follow the token role, so a head can fit them (`with_tokens_np` adds the loader's `tokens_np`)"""
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


def with_tokens_np(batch):
    """a synthetic batch + `tokens_np` as the loader makes it (float64 pos / dirv, e_p / flags_p / s_p = the sigmoid of the logits, cell, valid): what `rule_teacher` reads"""
    sg = lambda k: batch[k].float().sigmoid().double().numpy()  # noqa: E731
    pos, dirv, valid = batch["pos"].double().numpy(), batch["dirv"].double().numpy(), batch["valid"].numpy()
    return dict(batch, tokens_np=[dict(pos=pos[b], dirv=dirv[b], e_p=sg("e")[b], flags_p=sg("flags")[b], s_p=sg("s")[b], cell=np.arange(pos.shape[1]), valid=valid[b].copy()) for b in range(len(pos))])
