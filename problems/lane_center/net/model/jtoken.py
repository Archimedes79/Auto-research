"""E365 (stage S1 of the token champion, D113): the E2E token head on the PREDICTED nodes of the frozen lane head B'; the wiring is the per-source successor softmax with a none slot.

    head = JTokenHead()                                       # 0.127 M parameters; the buffer `step` counts the training forwards (the switch of the key weights, below)
    out = head(batch)                                         # batch of `net/prep/lane_ptok_data.loader` (section 2a of the card); only the INPUT keys are read, never a key that carries GT
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
E395 (`JTokenHead(pos_head=True)`, default off = the head above bit for bit: no extra parameter, no state key, no output key): a position head `posh` (MLP d -> d -> 1, GELU, last Linear at zero), built LAST, reads the final features
(the LayerNorm(x) of the wiring read): `out["dpos"] = POS_CAP tanh(posh / POS_CAP)` (B, K) m, the offset of a token along its normal n = rot90(dirv) (`refined_pos`: pos + dpos n; pos and dirv are the head's inputs, cut from the graph).
`jtoken_loss(curve_w=W, node_w=V)` (default 0 = nothing read, nothing added): + W x the Catmull-Rom curve loss and / or + V x the node loss of `curve_losses`, on the GT geometry keys of the batch (`GEOM_KEYS`, loader geom=True,
`lane_gtgeom_data`): the lateral distance, smooth L1 (knee 0.5 m), between the curve through the refined positions of the best tokens of the GT nodes and the curve through the GT nodes (uniform Catmull-Rom, `cr_points_t`), per GT link at
5 values of t, ghost nodes from the GT neighbours (reflection when one is missing, in both curves), links with a GT chord above 8 m or unequal GT chords (ratio above 2) left out and counted; the node loss is the same distance of each node alone. The forward never reads a GT key. Tests: test_jtoken.py
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
for sub_ in (".", "../prep", "../loss"):
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, sub_)))
import jtok as JT  # noqa: E402

SHORT, NG = JT.SHORT, JT.NG
NEG = -1e4                                                                  # the fill of a masked logit
SKIP_KEYS = ("sigma", "hops", "m", "t_short", "t_long", "keep")             # E399: the settings of the soft skip-link layer (`skip_suppress`)
GAP_KEYS = ("sigma", "sigma_lo", "gamma", "delta", "lo", "lo_long", "t_short", "t_long", "keep")    # E420: the settings of the soft gap-closing layer (`close_gaps`)
TOK, D, NTYPE = 64, 64, 7                                                   # B's node token, the width, the node types (0 none, 1 lane point, 2 fork, 3 merge, 4 entry, 5 exit, 6 boundary)
SWITCH = 1000                                                               # training steps with the key weight sigmoid(e) in every layer; then the head's own 1 - p_none
BANDS = 4
NIN = 2 + 4 * BANDS + 2 + 1 + 1 + 8 + 3                                     # pos / HALF, its Fourier features, dirv, e, e_raw, 8 flags, s, uout, uin
INPUTS = ("tok", "pos", "dirv", "e", "e_raw", "flags", "s", "uout", "uin", "valid")      # the only batch keys the forward reads
GT_KEYS = ("y", "ylong", "yscale", "ytype")                                 # the keys of the batch that carry GT (the loss reads them, the forward never)
EXCHANGE = ("real", "uniform")                                              # E392: the weights of a message round (`message_weights`): the early read's own probabilities, or the content null
SUPERVISE = ("none", "self")                                                # the early reads only supervise (training mode only): p_k does not enter x
MSG = EXCHANGE + SUPERVISE                                                  # + none: no exchange, no round module (deep supervision); self: the round block on the token's own features (the capacity null)
POS_CAP = 3.0                                                               # E395: the largest normal offset of the position head (m)
def _const(t):
    """a cached input: no gradient into B' (the one place that cuts the graph; skipped under `live_inputs`)"""
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


def link_prob(logp, cand):
    """logp (B, 2, K, K + 1) of `wiring_logp`, cand (B, 2, K, K) -> (B, K, K) the probability of the link i -> j in the class of the pair (0 outside the gate)"""
    return (logp[..., :cand.shape[-1]].exp() * cand).sum(1)


def skip_suppress(logp, cand, pnone, chord, cfg):
    """E399: the soft skip-link layer (no parameter, differentiable). logp (B, 2, K, K + 1) of `wiring_logp`, cand (B, 2, K, K), pnone (B, K) the none probability of the node type, chord (B, K, K) in m; cfg: sigma, hops, m, t_short,
    t_long, keep (the thresholds of the read-out). The soft "delivered" strength of a link is q = sigmoid((p - t) / sigma) x k_i x k_j with k = sigmoid((keep - pnone) / sigma) and t the threshold of the pair's class (chord above 8 m: long);
    R(i, j) = the largest product of q along a path i -> .. -> j of 2 .. hops links over every node's m strongest links; p'(i -> j) = p x (1 - R) and the removed mass goes to the source's none slot of the class, so every row stays a
    probability distribution. A link that another path of the same graph already covers (a jump ahead on its own lane) is thus damped"""
    B, K = cand.shape[0], cand.shape[-1]
    p = logp[..., :K].exp() * cand
    t = torch.where(chord > SHORT, float(cfg["t_long"]), float(cfg["t_short"])).to(p.dtype)
    s = float(cfg["sigma"])
    k = torch.sigmoid((float(cfg["keep"]) - pnone.to(p.dtype)) / s)
    q = torch.sigmoid((p.sum(1) - t) / s) * cand.any(1) * k[:, :, None] * k[:, None, :]
    w, idx = q.topk(min(int(cfg["m"]), K), dim=-1)                          # every node's strongest links
    m = idx.shape[-1]
    hop = lambda reach: (w[..., None] * reach.gather(1, idx.reshape(B, K * m, 1).expand(-1, -1, K)).view(B, K, m, K)).amax(2)  # noqa: E731
    reach = q                                                               # paths of 1 .. h links
    for _ in range(int(cfg["hops"]) - 2):
        reach = torch.maximum(q, hop(reach))
    p2 = p * (1.0 - hop(reach))[:, None]                                    # paths of 2 .. hops links
    lp = torch.where(cand, p2.clamp_min(1e-30).log(), logp[..., :K])
    none = (1.0 - p2.sum(-1)).clamp_min(1e-30).log()
    return torch.cat([lp, none[..., None]], -1)


def close_gaps(logp, cand, pnone, chord, cfg):
    """E420: the soft gap-closing layer (no parameter, differentiable). Arguments as `skip_suppress`; cfg: sigma, sigma_lo, gamma, delta, lo, lo_long, t_short, t_long, keep. With P the probability of a pair, t its class threshold, k the soft
    "kept" of a node and q = sigmoid((P - t) / sigma) x k_i x k_j the soft "delivered" strength of a link: a pair is OPEN by (1 - max_j' q(i, j')) x (1 - max_i' q(i', j)) (its source has no delivered successor and its target no
    delivered predecessor), ELIGIBLE by sigmoid((P - lo) / sigma_lo) x k_i x k_j (lo_long for the long class) and BEST by (m / max over the source's pairs)^gamma x (m / max over the target's pairs)^gamma with m = P x open x eligible
    (near 1 for the strongest such pair of both its ends, near 0 for any weaker one). P is lifted by open x eligible x best x max(t + delta - P, 0), the lifted mass is taken from the source's none slot of the class and, where that
    does not suffice, from its other links in proportion; every row stays a probability distribution. A weak link that is the only way to continue an open end is thus delivered"""
    K = cand.shape[-1]
    p = logp[..., :K].exp() * cand
    P = p.sum(1)
    is_long = chord > SHORT
    t = torch.where(is_long, float(cfg["t_long"]), float(cfg["t_short"])).to(P.dtype)
    lo = torch.where(is_long, float(cfg["lo_long"]), float(cfg["lo"])).to(P.dtype)
    s, gamma = float(cfg["sigma"]), float(cfg["gamma"])
    k = torch.sigmoid((float(cfg["keep"]) - pnone.to(P.dtype)) / s)
    ends = cand.any(1) * k[:, :, None] * k[:, None, :]
    q = torch.sigmoid((P - t) / s) * ends
    gate = (1.0 - q.amax(2))[:, :, None] * (1.0 - q.amax(1))[:, None, :] * torch.sigmoid((P - lo) / float(cfg["sigma_lo"])) * ends
    m = P * gate
    rel = lambda dim: (m / m.amax(dim, keepdim=True).clamp_min(1e-12)) ** gamma  # noqa: E731
    lift = (gate * rel(2) * rel(1) * (t + float(cfg["delta"]) - P).clamp_min(0.0))[:, None] * cand
    room = (1.0 - lift.sum(-1, keepdim=True)).clamp_min(0.0)
    p2 = p * (room / p.sum(-1, keepdim=True).clamp_min(1e-30)).clamp(max=1.0) + lift
    lp = torch.where(cand, p2.clamp_min(1e-30).log(), logp[..., :K])
    none = (1.0 - p2.sum(-1)).clamp_min(1e-30).log()
    return torch.cat([lp, none[..., None]], -1)


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
    `n_early` = the early reads after layer 1 .. rounds, `rounds` = the Round modules (none for msg 'none'); pos_head: E395, the position head `posh` is built after everything else (the same rule)"""

    def __init__(self, d=D, heads=4, layers=3, ffn=128, dq=32, live_inputs=False, rounds=0, msg="real", pos_head=False, skip=None, min_e=None, gap=None):
        super().__init__()
        assert 0 <= rounds < layers and msg in MSG, (rounds, layers, msg)
        assert skip is None or sorted(skip) == sorted(SKIP_KEYS), (skip, SKIP_KEYS)
        assert gap is None or sorted(gap) == sorted(GAP_KEYS), (gap, GAP_KEYS)
        assert min_e is None or 0.0 <= min_e < 1.0, min_e
        self.min_e = min_e                                                   # E399: only the tokens with existence sigmoid(e) above this take part (None = every valid token)
        self.dq = dq
        self.skip = None if skip is None else dict(skip)                     # E399: the soft skip-link layer on the final wiring (no parameter); None = off
        self.gap = None if gap is None else dict(gap)                        # E420: the soft gap-closing layer, after the skip layer (no parameter); None = off
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
        self.pos_head, self.pos_cap = pos_head, POS_CAP                     # E395: no module, no state key, no output key when off
        if pos_head:
            self.posh = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            nn.init.zeros_(self.posh[2].weight)                             # the head starts at the B' position (dpos = 0)
            nn.init.zeros_(self.posh[2].bias)
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
        thin = None
        if self.min_e is not None:                                          # E399: token thinning: the weak tokens leave attention, candidates and the graph
            thin = valid & (w_in <= self.min_e)
            valid = valid & ~thin
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
        if thin is not None:                                                # E399: a thinned token reads as none (the read-out keeps no such node)
            none = torch.zeros(NTYPE, device=types3.device, dtype=types3.dtype)
            none[0] = 30.0
            types3 = torch.where(thin[..., None], none, types3)
            w3 = torch.where(thin, torch.zeros_like(w3), w3)
        if wb is None:
            wb = self.wbias(g).float().permute(0, 3, 1, 2)
        logp, xn = self.read_wiring(x, wb, lw(w3 if learned else w_in), cand)          # the wiring layer
        if self.skip is not None:                                           # E399
            logp = skip_suppress(logp, cand, types3.softmax(-1)[..., 0], chord, self.skip)
        if self.gap is not None:                                            # E420
            logp = close_gaps(logp, cand, types3.softmax(-1)[..., 0], chord, self.gap)
        scale = 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[:, :, None] + self.sc_j(xn)[:, None] + self.sc_g(g))).float().sigmoid()
        if self.training:
            self.step += 1
            self._step_py += 1
        out = dict(logp=logp, cand=cand, scale=scale, types=types3, types1=types1, valid=valid, chord=chord)
        if early:
            out["logp_rounds"] = early
        if self.pos_head:                                                   # E395: the normal offset of every token, from the final features only
            out["dpos"] = self.pos_cap * torch.tanh(self.posh(xn)[..., 0].float() / self.pos_cap)
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------------------------------------------------------------------ a synthetic batch
def _toy_geom(rng, pos, dirv):
    """the GT geometry keys of `synthetic_batch(geom=True)` (a consistent toy): GT node g (0 .. 29) = lane token g moved sideways by up to 0.5 m (a refinement has something to learn), its best token = token g (GT node 5 has none),
    the links of the chain 0 .. 19 (lanes 0 and 1, joined by the long link 9 -> 10) and of the chain 20 .. 29 with the GT neighbours as ghosts (-1 at the ends); padded to 32 nodes and 30 links"""
    B, Ng, Lg = len(pos), 32, 30
    gt_pos, gt_dir, gt_btok, gl = np.zeros((B, Ng, 2), np.float32), np.zeros((B, Ng, 2), np.float32), -np.ones((B, Ng), np.int64), -np.ones((B, Lg, 4), np.int64)
    off = rng.uniform(-0.5, 0.5, (B, 30)).astype(np.float32)
    gt_pos[:, :30] = pos[:, :30] + off[..., None] * np.stack([-dirv[:, :30, 1], dirv[:, :30, 0]], -1)
    gt_dir[:, :30], gt_btok[:, :30], gt_btok[:, 5] = dirv[:, :30], np.arange(30), -1
    links = [(g, g + 1, -1 if g in (0, 20) else g - 1, -1 if g + 1 in (19, 29) else g + 2) for g in list(range(19)) + list(range(20, 29))]
    gl[:, :len(links)] = links
    return dict(gt_pos=torch.from_numpy(gt_pos), gt_dir=torch.from_numpy(gt_dir), gt_btok=torch.from_numpy(gt_btok), gl=torch.from_numpy(gl))


def synthetic_batch(B=8, K=48, seed=0, geom=False):
    """random frames that follow the batch dict of the card (2a): per frame a lane of 10 tokens 2 m apart, a second lane 12 m behind its end (one long link), a third lane elsewhere, 4 duplicate tokens 0.5 m beside
    a lane token (they are in G(i) of its predecessor, assigned to no GT node), padding to K. Types and flags follow the token role, so a head can fit them. geom (E395): + the GT geometry keys `GEOM_KEYS` (`_toy_geom`, drawn after
    everything else: the other keys are those of geom False)"""
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
    batch = dict(tok=torch.from_numpy(r(B, K, TOK) * valid[..., None]), pos=torch.from_numpy(pos), dirv=torch.from_numpy(dirv), e=torch.from_numpy(e), e_raw=torch.from_numpy(e + 0.3 * r(B, K)),
                 flags=torch.from_numpy(flags), s=torch.from_numpy(r(B, K)), uout=torch.from_numpy(r(B, K)), uin=torch.from_numpy(r(B, K)), valid=torch.from_numpy(valid), ytype=torch.from_numpy(ytype),
                 y=torch.from_numpy(y), ylong=torch.from_numpy(ylong), yscale=torch.from_numpy(yscale))
    if geom:
        batch.update(_toy_geom(rng, pos, dirv))
    return batch
