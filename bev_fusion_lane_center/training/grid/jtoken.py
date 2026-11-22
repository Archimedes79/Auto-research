"""E365 (stage S1 of the token champion, D113): the E2E token head on the PREDICTED nodes of the frozen lane head B'; the wiring is the per-source successor softmax with a none slot.

    head = JTokenHead()                                       # 0.127 M parameters; the buffer `step` counts the training forwards (the switch of the key weights, below)
    out = head(batch)                                         # batch of `data/lane_ptok_data.loader` (section 2a of the card); only the INPUT keys are read, never a key that carries GT
    loss, info = jtoken_loss(out, batch)                      # the exact likelihood below + type CE + scale L1 (reads y, ylong, yscale, ytype)
    pr = probs(out); links = decode(out, thr=0.3)             # decode of record; decode_argmax(out) = the alternative (one link per source and class)
    batch = synthetic_batch(B, K, seed)                       # a random batch that follows the batch dict (tests and the trainer's --smoke)

Input keys (B frames, K tokens): tok (B,K,64), pos (B,K,2) m, dirv (B,K,2) unit, e / e_raw (B,K) logits, flags (B,K,8) logits, s / uout / uin (B,K) logits, valid (B,K) bool. All of them are constants (the cached B' is
frozen: the forward cuts the graph at them, no gradient reaches B' and none goes through the hard top-K). S4 (E372) `JTokenHead(live_inputs=True)`: the forward keeps their graph (a live B' in the training step gets the gradient
of the head's loss); the values, and so every output, are the same.
Embedding: x0 = LN(Linear(tok) + MLP[Fourier(pos), dirv, e, e_raw, flags, s, uout, uin]). 3 pre-norm self-attention layers (4 heads, FFN 128): the logit of head h = q.k / 4 + bias_h(g_ij) + log(w_j + 1e-3), bias = an MLP
13 -> 32 -> H of the 13 `jtok.pair_geometry` features (the existence enters them as sigmoid(e)), padding tokens masked as keys.
Soft pruning: the key weight w_j is sigmoid(e_j) (input) in layer 1 and for the first SWITCH training steps in every layer; afterwards w_j = 1 - p_none(j) of the head's OWN type softmax (7 types, type 0 = none), read by
the SAME Linear (after a shared LayerNorm) after layer 1 (keys of layers 2 and 3) and after layer 3 (keys of the wiring layer); both readings get the type CE, and the key-bias path carries gradient into them.
Wiring layer (on top of layer 3, attention weights only): class c in {short: chord <= 8 m, long: chord > 8 m}, z^c_ij = q^c_i.k^c_j / sqrt(32) + bias^c(g_ij) + log(w_j + 1e-3) + b^c_j on the allowed gate of `jtok.pair_geometry`
(padding masked as key and source) + ONE learned none slot per class (key k^c_none and a bias): p^c(.|i) = softmax over {allowed keys} + {none}. Two handle scales per pair from an MLP of (x_i, x_j, g_ij).
Likelihood (`jtoken_loss`): per class c and source i (every valid token with at least one candidate of the class): G(i) = the in-gate tokens with y[i, j] = 1 whose pair lies in the class by the TOKEN chord (ylong is only a statistic and the long scale target), nll_i = -log sum_{j in G(i)} p^c(j|i)
(E376 `pos_loss="max"`: -log max_{j in G(i)} p^c(j|i)) when G(i) is non-empty, -log p^c(none|i) when it is empty (also when its GT links lie outside the gate: counted in info); per frame the mean over the sources of the class, then over the frames. + 0.5 x (type CE after layer 1
+ type CE after layer 3, per frame the mean over the valid tokens) + 0.5 x L1 on the handle scales of the long GT links. Tests: test_jtoken.py
E379 (`distill_loss`, ensemble distillation; the trainer adds W x it to the loss above): p_T = the MEAN of the frozen teachers' PROBABILITIES on the same tokens (`teacher_logp`); kd = KL_s + KL_l, the KL(p_T || p) of the
successor softmax (with the none slot) per class over the sources of the likelihood (`source_rows`), temperature 1; the type readings and the handle scales are not distilled (card v2).
E392 (`JTokenHead(rounds=R, msg=)`, default 0 = the head above bit for bit: no extra parameter, no extra output key): after layer 1 (R >= 1) and after layer 2 (R = 2) a round reads the wiring from the current features with the SAME weights as the final
read (`read_wiring`; key weight of layers 2 - 3), p_k = its probability of the link i -> j summed over the two classes, and gives every token the p_k-weighted features of its likely predecessors and successors, x <- x + MLP([mf ; mb]) (`Round`, own
weights per round, last layer at zero: a rounds head starts as the champion function; p_k not detached); `types1` / w1 are read BEFORE round 1; the early reads are `out["logp_rounds"]`. msg='uniform' (the content null, `message_weights`): the same degree mass spread uniformly over the allowed pairs. `jtoken_loss(wire_w=W)`: + W x the likelihood of every early read.
msg='none' (deep supervision without exchange): the same early reads and `out["logp_rounds"]`, but nothing is added to x and no `Round` exists: the parameters, the state keys and their order are the champion's (127131), and the early reads run in TRAINING mode only: in eval mode
the forward is the champion's bit for bit (no early read, no `logp_rounds`), so a head trained with it is at inference the unchanged champion network.
msg='self' (the capacity null of the exchange): the same `Round` modules (the parameters of the real rounds) but every token's message is its OWN features, x <- x + MLP([W_v x ; W_u x]): no neighbour is read, the early reads only supervise
(training mode only, so p_k does not enter x), the rounds act in eval mode too. With wire_w 0 it is the block alone, with wire_w > 0 the block after a supervised read.
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
NEG = -1e4                                                                  # the fill of a masked logit
TOK, D, NTYPE = 64, 64, 7                                                   # B's node token, the width, the node types (0 none, 1 lane point, 2 fork, 3 merge, 4 entry, 5 exit, 6 boundary)
SWITCH = 1000                                                               # training steps with the key weight sigmoid(e) in every layer; then the head's own 1 - p_none
BANDS = 4
NIN = 2 + 4 * BANDS + 2 + 1 + 1 + 8 + 3                                     # pos / HALF, its Fourier features, dirv, e, e_raw, 8 flags, s, uout, uin
INPUTS = ("tok", "pos", "dirv", "e", "e_raw", "flags", "s", "uout", "uin", "valid")      # the only batch keys the forward reads
GT_KEYS = ("y", "ylong", "yscale", "ytype")                                 # the keys of the batch that carry GT (the loss reads them, the forward never)
POS_LOSS = ("sum", "max")                                                   # the likelihood of a source with GT successors (`_source_nll`): -log sum over its group (default) or -log max (E376)
EXCHANGE = ("real", "uniform")                                              # E392: the weights of a message round (`message_weights`): the early read's own probabilities, or the content null
SUPERVISE = ("none", "self")                                                # the early reads only supervise (training mode only): p_k does not enter x
MSG = EXCHANGE + SUPERVISE                                                  # + none: no exchange, no round module (deep supervision); self: the round block on the token's own features (the capacity null)


def _const(t):
    """a cached input: no gradient into B' (the one place that cuts the graph; skipped under `live_inputs`)"""
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


def link_prob(logp, cand):
    """logp (B, 2, K, K + 1) of `wiring_logp`, cand (B, 2, K, K) -> (B, K, K) the probability of the link i -> j in the class of the pair (0 outside the gate)"""
    return (logp[..., :cand.shape[-1]].exp() * cand).sum(1)


def message_weights(p, cand, msg="real"):
    """E392: the weights of one message round from p (B, K, K) = the early probability of the link i -> j (`link_prob`) and the gate cand (B, 2, K, K) -> (w_fwd, w_bwd) (B, K, K): token j gets the forward message sum_i w_fwd[i, j] W_v x_i (its likely
    predecessors), token i the backward message sum_j w_bwd[i, j] W_u x_j (its likely successors). 'real': both are p. 'uniform' (the content null): each token keeps its soft degree (indeg_j = sum_i p[i, j], outdeg_i = sum_j p[i, j]) but spreads it
    uniformly over its allowed pairs: w_fwd[i, j] = gate[i, j] / sum_i' gate[i', j] x indeg_j, w_bwd[i, j] = gate[i, j] / sum_j' gate[i, j'] x outdeg_i (an empty sum: weight 0); no information about WHICH neighbours the early read found likely"""
    assert msg in EXCHANGE, msg
    if msg == "real":
        return p, p
    gate = cand.sum(1).float()
    return gate / gate.sum(1, keepdim=True).clamp_min(1) * p.sum(1, keepdim=True), gate / gate.sum(2, keepdim=True).clamp_min(1) * p.sum(2, keepdim=True)


class Round(nn.Module):
    """E392: one message round, x <- x + MLP([mf ; mb]) (no LayerNorm) with mf_j = sum_i w_fwd[i, j] W_v x_i and mb_i = sum_j w_bwd[i, j] W_u x_j (`message_weights`); the weights are not detached (the final loss reaches the early read through
    them). The LAST linear layer of the MLP starts at zero (weight and bias): a new round is the identity, so a rounds head starts as the champion function; its first optimiser step moves the last layer only, the gradient reaches the rest after it"""

    def __init__(self, d=D):
        super().__init__()
        self.wv, self.wu = nn.Linear(d, d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x, w_fwd, w_bwd):
        """w_fwd / w_bwd None (E392 msg 'self', the capacity null): every token's message is its own features, the same as the identity weights"""
        v, u = self.wv(x), self.wu(x)
        mf, mb = (v, u) if w_fwd is None else (w_fwd.transpose(1, 2).to(v.dtype) @ v, w_bwd.to(u.dtype) @ u)
        return x + self.mlp(torch.cat([mf, mb], -1))


class JTokenHead(nn.Module):
    """live_inputs: False = the INPUTS are cut from their graph (`_const`), True = kept (S4); rounds / msg: E392 (module docstring), the round modules are built LAST (a seeded init of the head's other parameters is the same as without them);
    `n_early` = the early reads after layer 1 .. rounds, `rounds` = the Round modules (none for msg 'none')"""

    def __init__(self, d=D, heads=4, layers=3, ffn=128, dq=32, live_inputs=False, rounds=0, msg="real"):
        super().__init__()
        assert 0 <= rounds < layers and msg in MSG, (rounds, layers, msg)
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
        self.n_early, self.msg = rounds, msg                                # E392
        self.rounds = nn.ModuleList([Round(d) for _ in range(0 if msg == "none" else rounds)])      # empty at rounds = 0 and for msg 'none': no parameter, no state key
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self._step_py = 0                                                   # the Python mirror of `step` (no device sync per forward); kept by forward, set_step and load_state_dict
        self.live_inputs = live_inputs

    def set_step(self, n):
        self.step.fill_(n)
        self._step_py = int(n)

    def _load_from_state_dict(self, *a, **k):
        super()._load_from_state_dict(*a, **k)
        self._step_py = int(self.step)

    def read_types(self, x):
        """-> (type logits, 1 - p_none)"""
        t = self.types(self.tnorm(x)).float()
        return t, 1.0 - t.softmax(-1)[..., 0]                               # logits, 1 - p_none

    def read_wiring(self, x, wb, logw, cand):
        """the wiring read of the module docstring on the features x (B, K, d): wb (B, 2, K, K) = bias^c(g_ij), logw (B, K) = the log key weight, cand the gate -> (the log-probabilities of `wiring_logp`, LayerNorm(x)); the final read and every
        early read of a round (E392) are THIS function with the head's own weights, so they are one reading"""
        B, K, _ = x.shape
        xn = self.lnw(x)
        q, k = (m(xn).view(B, K, 2, self.dq) for m in (self.wq, self.wk))
        z = torch.einsum("bicd,bjcd->bcij", q, k).float() / math.sqrt(self.dq) + wb + logw[:, None, None, :] + self.dest(xn).float().permute(0, 2, 1)[:, :, None, :]
        zn = torch.einsum("bicd,cd->bci", q, self.none_k).float() / math.sqrt(self.dq) + self.none_b[None, :, None]
        return wiring_logp(z, zn, cand), xn

    def exchange(self, k, x, wb, logw, cand):
        """E392 round k: the early read of x (`read_wiring`) and the messages along it (`Round`) -> (the new x, the early log-probabilities); msg 'none': x itself (the read only supervises); msg 'self': the round on the token's own features"""
        logp, _ = self.read_wiring(x, wb, logw, cand)
        if self.msg == "none":
            return x, logp
        if self.msg == "self":
            return self.rounds[k](x, None, None), logp
        return self.rounds[k](x, *message_weights(link_prob(logp, cand), cand, self.msg)), logp

    def forward(self, batch):
        b = {k: batch[k] if self.live_inputs else _const(batch[k]) for k in INPUTS}
        valid, pos, dirv = b["valid"].bool(), b["pos"].float(), b["dirv"].float()
        w_in = b["e"].float().sigmoid()
        g, allowed = JT.pair_geometry(pos, dirv, valid, e=w_in)
        lw = lambda w: (w + 1e-3).log()  # noqa: E731
        learned = self._step_py >= SWITCH
        chord = g[..., 2] * 20.0
        cand = torch.stack([allowed & (chord <= SHORT), allowed & (chord > SHORT)], 1)
        n_early = self.n_early if self.msg not in SUPERVISE or self.training else 0     # msg 'none' / 'self': the early reads only supervise, so they exist in training mode only (none: inference = the champion's forward)
        wb = self.wbias(g).float().permute(0, 3, 1, 2) if n_early else None         # bias^c(g_ij), the same for every read (no early read: made at the final read, where the champion made it: the backward order into g stays the champion's)
        feat = torch.cat([pos / JT.HALF, fourier(pos), dirv, b["e"][..., None], b["e_raw"][..., None], b["flags"], b["s"][..., None], b["uout"][..., None], b["uin"][..., None]], -1).float()
        x = self.ln0(self.proj(b["tok"].float()) + self.mlp(feat))
        x = self.layers[0](x, g, lw(w_in), valid)
        types1, w1 = self.read_types(x)
        early = []                                                          # E392: the early reads
        for n, layer in enumerate(self.layers[1:], 1):
            if n <= n_early:                                                # a round after layer n, before the next one
                x, logp_k = self.exchange(n - 1, x, wb, lw(w1 if learned else w_in), cand)
                early.append(logp_k)
            elif self.msg == "self" and n <= self.n_early:                  # msg 'self' in eval mode: the round block runs, the read that only supervised does not
                x = self.rounds[n - 1](x, None, None)
            x = layer(x, g, lw(w1 if learned else w_in), valid)
        types3, w3 = self.read_types(x)
        if wb is None:
            wb = self.wbias(g).float().permute(0, 3, 1, 2)
        logp, xn = self.read_wiring(x, wb, lw(w3 if learned else w_in), cand)          # the wiring layer
        scale = 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[:, :, None] + self.sc_j(xn)[:, None] + self.sc_g(g))).float().sigmoid()
        if self.training:
            self.step += 1
            self._step_py += 1
        out = dict(logp=logp, cand=cand, scale=scale, types=types3, types1=types1, valid=valid, chord=chord)
        if early:
            out["logp_rounds"] = early
        return out


# ------------------------------------------------------------------------------------------------------------------------------ probabilities and decodes
def probs(out):
    """-> p (B, K, K) the probability of the link i -> j in the class of the pair (0 outside the gate), pnone (B, K, K) the none probability of the source i in the class of the pair, pt (B, K, 7) the type probabilities,
    keep (B, K) the kept nodes (valid and p_none < 0.5, the output rule), cand (B, K, K) the allowed gate, longm (B, K, K) chord above 8 m"""
    K = out["cand"].shape[-1]
    cand = out["cand"]
    p = link_prob(out["logp"], cand)
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


def source_rows(out):
    """(B, 2, K) bool: the sources of a class, the valid tokens with a candidate of it (the rows of the likelihood and of the KD term)"""
    return out["cand"].any(-1) & out["valid"][:, None]


def jtoken_loss(out, batch, type_w=0.5, scale_w=0.5, none_weight=1.0, pos_loss="sum", wire_w=0.0):
    """the likelihood of the module docstring -> (loss, info); batch: y (B, K, K), ylong (B, K, K), yscale (B, K, K, 2), ytype (B, K). wire_w (E392, default 0 = nothing added, `logp_rounds` not read): + wire_w x (nll_s + nll_l) of every early read
    of `out["logp_rounds"]` (the same `_source_nll`, none weight, pos_loss, rows and per-frame normalisation as the final one), in info nll1_s, nll1_l, nll2_s, nll2_l. info: nll_s, nll_l, ce1, ce3, scale_l1 and the counts gt_links, pos_pairs (in-gate GT
    links), gt_outside (links with no gate), gt_class (in the gate, but `ylong` disagrees with the token chord: a statistic, the class of a link is the token chord's). The long scale target is the fitted yscale
    of the long-class links (finite where the cache fitted one). none_weight W (default 1: the loss of before): multiplies (i) the loss of the sources whose G(i) is empty (their target is the none slot), in both classes, and (ii) the type
    cross-entropy of the tokens whose type is none (class 0 weight W, the six others 1, normalised by the sum of the weights of the valid tokens of the frame, as torch's weighted mean). pos_loss 'sum' (default: the loss of
    before) or 'max' (E376): the likelihood of the sources with a non-empty G(i), in both classes (`_source_nll`); the empty-G terms, the type cross-entropy and the scale term are the same under both.
    scale_w = 0: no scale term, `yscale` is not read"""
    dev, cand, lp, valid = out["logp"].device, out["cand"], out["logp"], out["valid"]
    y, yl = batch["y"].to(dev) > 0, batch["ylong"].to(dev).bool()
    G = y[:, None] & cand                                                  # (B, 2, K, K): the in-gate GT successors of each class (the class of a pair = its token chord)
    yt = batch["ytype"].to(dev).long()
    nll = _source_nll(lp, G, none_weight, pos_loss)
    rows = source_rows(out)
    nll_s, nll_l = (_frame_mean(nll[:, c], rows[:, c]) for c in range(2))
    wt = torch.where(yt == 0, none_weight, 1.0) if none_weight != 1.0 else None
    ce1, ce3 = (_frame_mean(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt) * (1.0 if wt is None else wt), valid, wt) for t in (out["types1"], out["types"]))
    loss = nll_s + nll_l + type_w * (ce1 + ce3)
    info = dict(nll_s=float(nll_s), nll_l=float(nll_l), ce1=float(ce1), ce3=float(ce3), gt_links=float(y.sum()), pos_pairs=float(G.sum()), gt_outside=float((y & ~cand.any(1)).sum()),
                gt_class=float((y & cand.any(1) & (yl != (out["chord"] > SHORT))).sum()))
    if wire_w:                                                              # E392: the same likelihood on every early read
        assert "logp_rounds" in out, "wire_w > 0 needs a head with rounds, in training mode for msg 'none' (the output has no early read)"
        for k, lpk in enumerate(out["logp_rounds"], 1):
            nk = _source_nll(lpk, G, none_weight, pos_loss)
            s_k, l_k = (_frame_mean(nk[:, c], rows[:, c]) for c in range(2))
            loss = loss + wire_w * (s_k + l_k)
            info.update({f"nll{k}_s": float(s_k), f"nll{k}_l": float(l_k)})
    if not scale_w:
        return loss, info
    ys = batch["yscale"].to(dev).float()
    m = G[:, 1] & torch.isfinite(ys[..., 0])
    if m.any():
        l1 = (out["scale"].float()[m] - ys[m]).abs().mean()
        loss = loss + scale_w * l1
        info["scale_l1"] = float(l1)
    return loss, info


def teacher_logp(lps):
    """the log of the MEAN of the teachers' probabilities (not of their logits): lps = their log-probabilities (..., C) -> (..., C); one teacher -> its own bit for bit; a key that every teacher masks (log p -inf or the
    NEG fill) stays at p 0"""
    return torch.logsumexp(torch.stack(lps), 0) - math.log(len(lps))


def _kl(lpt, lp):
    """KL(p_T || p) per row from log-probabilities: the sum over the last axis of p_T (log p_T - log p); an entry with p_T = 0 (a masked key) adds exactly 0 (no 0 x log 0, no 0 x -inf)"""
    pt = lpt.exp()
    return torch.where(pt > 0, pt * (lpt - lp), 0.0).sum(-1)


def distill_loss(out, outs_t):
    """E379 (module docstring): out = the student's forward, outs_t = the frozen teachers' forwards on the SAME batch (no gradient) -> (kd, info: kd, its parts kd_s, kd_l, and ent_s, ent_l = the entropy of p_T per source
    of the class). The trainer adds W x kd to the likelihood and checks at its first step that the teachers saw the student's tokens (the same cand and valid)"""
    assert outs_t, "no teacher"
    lpt = teacher_logp([o["logp"] for o in outs_t])                        # (B, 2, K, K + 1): the candidates + the none slot
    src = source_rows(out)
    kl, ent = _kl(lpt, out["logp"]), -_kl(lpt, torch.zeros_like(lpt))      # H(p_T) = -KL(p_T || 1)
    kd_s, kd_l = (_frame_mean(kl[:, c], src[:, c]) for c in range(2))
    kd = kd_s + kd_l
    return kd, dict(kd=float(kd), kd_s=float(kd_s), kd_l=float(kd_l), ent_s=float(_frame_mean(ent[:, 0], src[:, 0])), ent_l=float(_frame_mean(ent[:, 1], src[:, 1])))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------------------------------------------------------------------ a synthetic batch
def synthetic_batch(B=8, K=48, seed=0):
    """random frames that follow the batch dict of the card (2a): per frame a lane of 10 tokens 2 m apart, a second lane 12 m behind its end (one long link), a third lane elsewhere, 4 duplicate tokens 0.5 m beside
    a lane token (they are in G(i) of its predecessor, assigned to no GT node), padding to K. Types and flags follow the token role, so a head can fit them"""
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
