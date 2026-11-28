"""The token head: self-attention among the K tokens of the lane head B' (its predicted nodes); the wiring is a per-source successor softmax with a none
slot, the node type a seven-way softmax. Its loss is `net/loss/loss_jtoken.py`; `lane_net.build_head` builds it with the champion's sizes.

    head = JTokenHead()                       # the buffer `step` counts the training forwards (the switch of the key weights, below)
    out = head(batch)                         # reads only the INPUTS keys of a batch (`lane_ptok_data` / `lane_live_data`), never a key that carries GT
    pr = probs(out); links = decode(out)      # link probabilities and a plain decode (the delivered read-out is `lane_readout.py`)
    batch = synthetic_batch(B, K, seed)       # a random batch of the batch dict (tests, the trainer's --smoke)

Inputs (B frames, K tokens): tok (B, K, 64), pos (B, K, 2) m, dirv (B, K, 2) unit, e / e_raw (B, K) logits, flags (B, K, 8) logits, s / uout / uin
(B, K) logits, valid (B, K) bool. `live_inputs` False: the forward cuts their graph (`_const`: cached tokens); True: it keeps it (stage 2 trains B''s
node head through the head's loss); the values, and so every output, are the same.
Embedding: x0 = LN(Linear(tok) + MLP[Fourier(pos), dirv, e, e_raw, flags, s, uout, uin]). Three pre-norm self-attention layers (4 heads, FFN 128): the
logit of head h = q.k / 4 + bias_h(g_ij) + log(w_j + 1e-3), bias an MLP of the 13 pair features of `jtok.pair_geometry`, padding masked as keys.
Key weight w_j: sigmoid(e_j) in layer 1, and in every layer for the first SWITCH training steps; afterwards 1 - p_none(j) of the head's own type
softmax, read by one Linear after layer 1 and after layer 3 (both readings get the type loss).
Wiring (on top of layer 3): class c in {short: chord <= 8 m, long: chord > 8 m}, z^c_ij = q^c_i.k^c_j / sqrt(32) + bias^c(g_ij) + log(w_j + 1e-3)
+ b^c_j on the allowed pairs of `jtok.pair_geometry`, plus one learned none slot per class: p^c(.|i) = softmax over {allowed keys} + {none}. Two
handle scales per pair from an MLP of (x_i, x_j, g_ij).
Soft output layers (no parameter; off unless a run's args name them, `OUTPUT_LAYERS`), in this order after the wiring: `skip_suppress`, `close_gaps`,
`fork_gate`; `refine_positions` writes the delivered node positions `pos`. Tests: net/tests/test_jtoken.py"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lane_center.net.model import jtok as JT

SHORT, NG = JT.SHORT, JT.NG
NEG = -1e4                                                                  # the fill of a masked logit
SKIP_KEYS = ("sigma", "hops", "m", "t_short", "t_long", "keep")             # E399: the settings of the soft skip-link layer (`skip_suppress`)
GAP_KEYS = ("sigma", "sigma_lo", "gamma", "lo", "lo_long", "t_short", "t_long", "keep")    # E420: the settings of the soft gap-closing layer (`close_gaps`)
FORK_KEYS = ("sigma", "sigma_rel", "tau", "sigma_f", "t_short", "t_long", "keep")          # E423: the settings of the soft fork gate (`fork_gate`)
# the share a fully covered link keeps in `skip_suppress`: the loss of a head that trains with the layer stays finite
SKIP_FLOOR = 1e-3
# the largest share of a source's old mass `close_gaps` moves to its lifted links: at least half of the none slot stays
GAP_ROOM = 0.5
POS_KEYS = ("gain", "lam", "sigma", "t_short", "keep")                      # E424: the settings of the soft position layer (`refine_positions`)
# the side of B''s node cells (m): a node is its cell's centre plus an offset inside the cell
CELL = 2.0
# named sets of the soft output layers as a run's args hold them (`lane_net.build_head`), at the champion's read-out thresholds (0.3 / 0.2 / 0.6); the settings
# were chosen on the TRAIN sample (cards E399, E420, E423, E424)
OUTPUT_LAYERS = {"soft_output": dict(
    skip_suppress=dict(sigma=0.03, hops=5, m=8, t_short=0.3, t_long=0.2, keep=0.6),
    close_gaps=dict(sigma=0.01, sigma_lo=0.01, gamma=16.0, lo=0.03, lo_long=0.02, t_short=0.3, t_long=0.2, keep=0.6),
    fork_gate=dict(sigma=0.01, sigma_rel=0.02, tau=0.05, sigma_f=0.01, t_short=0.3, t_long=0.2, keep=0.6),
    refine_positions=dict(gain=1.2, lam=0.5, sigma=0.03, t_short=0.3, keep=0.6))}
# B's node token, the width, the node types (0 none, 1 lane point, 2 fork, 3 merge, 4 entry, 5 exit, 6 boundary)
TOK, D, NTYPE = 64, 64, 7
FORK = 2
# training steps with the key weight sigmoid(e) in every layer; then the head's own 1 - p_none
SWITCH = 1000
BANDS = 4
NIN = 2 + 4 * BANDS + 2 + 1 + 1 + 8 + 3                                     # pos / HALF, its Fourier features, dirv, e, e_raw, 8 flags, s, uout, uin
INPUTS = ("tok", "pos", "dirv", "e", "e_raw", "flags", "s", "uout", "uin", "valid")      # the only batch keys the forward reads
GT_KEYS = ("y", "ylong", "yscale", "ytype")                                 # the keys of the batch that carry GT (the loss reads them, the forward never)
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
    """z (B, 2, K, K) key logits, zn (B, 2, K) none logits, cand (B, 2, K, K) bool -> (B, 2, K, K + 1) float32 log-probabilities of the per-source softmax over
    {the candidates} + {none} (the LAST column); a key outside
    `cand` has probability exactly 0 and a source without a candidate has p(none) = 1"""
    return torch.cat([z.float().masked_fill(~cand, NEG), zn.float()[..., None]], -1).log_softmax(-1)


def link_prob(logp, cand):
    """logp (B, 2, K, K + 1) of `wiring_logp`, cand (B, 2, K, K) -> (B, K, K) the probability of the link i -> j in the class of the pair (0 outside the
    gate)"""
    return (logp[..., :cand.shape[-1]].exp() * cand).sum(1)


def skip_suppress(logp, cand, pnone, chord, cfg):
    """E399: the soft skip-link layer (no parameter, differentiable). logp (B, 2, K, K + 1) of `wiring_logp`, cand (B, 2, K, K), pnone (B, K) the none
    probability of the node type, chord (B, K, K) in m; cfg: sigma, hops, m, t_short, t_long, keep (the thresholds of the read-out). The soft "delivered"
    strength of a link is q = sigmoid((p - t) / sigma) x k_i x k_j with k = sigmoid((keep - pnone) / sigma) and t the threshold of the pair's class (chord above
    8 m: long);
    R(i, j) = the largest product of q along a path i -> .. -> j of 2 .. hops links over every node's m strongest links; p'(i -> j) = p x (1 - (1 - SKIP_FLOOR)
    R) and the removed mass goes to the source's none slot of the class, so every row stays a probability distribution. A link that another path of the same
    graph already covers (a jump ahead on its own lane) is thus damped"""
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
    p2 = p * (1.0 - (1.0 - SKIP_FLOOR) * hop(reach))[:, None]               # paths of 2 .. hops links
    lp = torch.where(cand, p2.clamp_min(1e-30).log(), logp[..., :K])
    none = (1.0 - p2.sum(-1)).clamp_min(1e-30).log()
    return torch.cat([lp, none[..., None]], -1)


def _max_of_others(q, dim):
    """for every entry of q the largest of the OTHER entries along dim (q.shape[dim] >= 2; of two equal largest entries each sees the other)"""
    v, i = q.topk(2, dim=dim)
    own = torch.zeros_like(q, dtype=torch.bool).scatter(dim, i.narrow(dim, 0, 1), True)
    return torch.where(own, v.narrow(dim, 1, 1), v.narrow(dim, 0, 1))


def close_gaps(logp, cand, pnone, chord, cfg):
    """E420: the soft gap-closing layer (no parameter, differentiable, never decreasing in a pair's own probability). Arguments as `skip_suppress`; cfg: sigma,
    sigma_lo, gamma, lo, lo_long, t_short, t_long, keep. With P the probability of a pair, t its class threshold, k the soft "kept" of a node and q = sigmoid((P
    - t) / sigma) x k_i x k_j the soft "delivered" strength of a link: a pair is OPEN by (1 - max_j' q(i, j')) x (1 - max_i' q(i', j)) over the OTHER pairs j'
    != j and i' != i (no other delivered link leaves its source or reaches its target), ELIGIBLE by sigmoid((P - lo) / sigma_lo) x k_i x k_j (lo_long for the
    long class) and BEST by (m / max over the source's pairs)^gamma x (m / max over the target's pairs)^gamma with m = P x open x eligible (1 for the strongest
    such pair of both its ends, near 0 for a clearly weaker one). With w = open x eligible x best, P is lifted by w x t x (1 - P): at w = 1 to at least t, above
    the threshold and still ordered by P. A pair's own P enters neither its openness nor anything that falls with it, so the output never falls when P rises. A
    row (source, class) whose lifts sum to L makes room by scaling ALL its old mass, the none slot included, by 1 - L' with L' = min(L, GAP_ROOM) and adds the
    lifts (scaled to the sum L'): every row stays a probability distribution and the none slot never empties (at least half of it stays: the loss of a head that
    trains with the layer stays finite). A weak link that is the only way to continue an open end is thus delivered"""
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
    gate = (1.0 - _max_of_others(q, 2)) * (1.0 - _max_of_others(q, 1)) * torch.sigmoid((P - lo) / float(cfg["sigma_lo"])) * ends
    m = P * gate
    rel = lambda dim: (m / m.amax(dim, keepdim=True).clamp_min(1e-12)) ** gamma  # noqa: E731
    lift = (gate * rel(2) * rel(1) * t * (1.0 - P).clamp_min(0.0))[:, None] * cand
    total = lift.sum(-1, keepdim=True)
    used = total.clamp(max=GAP_ROOM)
    p2 = p * (1.0 - used) + lift * (used / total.clamp_min(1e-12))
    lp = torch.where(cand, p2.clamp_min(1e-30).log(), logp[..., :K])
    none = logp[..., K] + torch.log1p(-used[..., 0])                           # the old none mass, scaled like the rest
    return torch.cat([lp, none[..., None]], -1)


def fork_gate(logp, cand, ptype, chord, cfg):
    """E423: the soft fork gate (no parameter, differentiable, never decreasing in a pair's own probability): a source takes a SECOND successor only as far as
    the head's own type softmax calls it a fork. ptype (B, K, 7); cfg: sigma, sigma_rel, tau, sigma_f, t_short, t_long, keep. With s_ij = P_ij / t (the
    probability over its class threshold, 0 unless both ends are kept) and m_ij = the largest s of the source's OTHER links, the link is a further link by w_ij
    =
    sigmoid((m_ij - s_ij) / sigma_rel) x sigmoid((m_ij - 1) / sigma_rel) (another link is stronger, and delivered); the source is a fork by f_i =
    sigmoid((p_fork(i) - tau) / sigma_f). p'(i -> j) = p x (1 - (1 - f_i) w_ij); the removed mass goes to the source's none slot of the class. The strongest
    link of a source and every link of a fork stay as they are"""
    K = cand.shape[-1]
    p = logp[..., :K].exp() * cand
    P = p.sum(1)
    t = torch.where(chord > SHORT, float(cfg["t_long"]), float(cfg["t_short"])).to(P.dtype)
    k = torch.sigmoid((float(cfg["keep"]) - ptype[..., 0].to(P.dtype)) / float(cfg["sigma"]))
    s = P / t * cand.any(1) * k[:, :, None] * k[:, None, :]
    m = _max_of_others(s, 2)
    rel = float(cfg["sigma_rel"])
    w = torch.sigmoid((m - s) / rel) * torch.sigmoid((m - 1.0) / rel)
    f = torch.sigmoid((ptype[..., FORK].to(P.dtype) - float(cfg["tau"])) / float(cfg["sigma_f"]))
    p2 = p * (1.0 - (1.0 - f[:, :, None]) * w)[:, None]
    lp = torch.where(cand, p2.clamp_min(1e-30).log(), logp[..., :K])
    none = (1.0 - p2.sum(-1)).clamp_min(1e-30).log()
    return torch.cat([lp, none[..., None]], -1)


def refine_positions(pos, dirv, valid, logp, cand, pnone, cfg):
    """E424: the soft position layer (no parameter, differentiable) against the zig-zag of the delivered lines -> positions (B, K, 2). cfg: gain, lam, sigma,
    t_short, keep.
    1. The node head's offset from its 2 m cell centre is shrunk toward the centre; it is scaled by `gain` and held inside the cell (+- 1 m).
    2. Along every short link, a pair of neighbours on one lane lies on a circular arc when the chord has the mean of the two headings: h = (d_i + d_j) /
       |d_i + d_j|, n = h turned left, r_ij = n . (pos_j - pos_i) is the sideways miss. Both ends move toward each other by half of it, weighted by the soft
       delivered strength q_ij = sigmoid((p - t_short) / sigma) k_i k_j, averaged over a node's links (not divided by a weight below 1), times `lam`. Straight
       lanes and circular arcs are fixed points: curves are not pulled straight"""
    half = float(JT.HALF)
    centre = (torch.floor((pos.detach() + half) / CELL) + 0.5) * CELL - half
    pos = torch.where(valid[..., None], centre + (float(cfg["gain"]) * (pos - centre)).clamp(-CELL / 2, CELL / 2), pos)
    K = cand.shape[-1]
    p = (logp[:, 0, :, :K].exp() * cand[:, 0]).to(pos.dtype)                  # the short class
    k = torch.sigmoid((float(cfg["keep"]) - pnone.to(pos.dtype)) / float(cfg["sigma"]))
    q = torch.sigmoid((p - float(cfg["t_short"])) / float(cfg["sigma"])) * cand[:, 0] * k[:, :, None] * k[:, None, :]
    h = F.normalize(dirv[:, :, None] + dirv[:, None], dim=-1)                  # (B, K, K, 2) the mean heading of i and j
    n = torch.stack([-h[..., 1], h[..., 0]], -1)
    res = (n * (pos[:, None] - pos[:, :, None])).sum(-1)                       # (B, K, K) n . (pos_j - pos_i)
    pull = (q * res)[..., None] * n * 0.5
    delta = pull.sum(2) - pull.sum(1)
    weight = q.sum(2) + q.sum(1)
    return pos + float(cfg["lam"]) * delta / weight.clamp_min(1.0)[..., None]


class JTokenHead(nn.Module):
    """live_inputs: False = the INPUTS are cut from their graph (`_const`), True = kept (S4)"""

    def __init__(self, d=D, heads=4, layers=3, ffn=128, dq=32, live_inputs=False, skip=None, gap=None, fork=None, refine=None):
        super().__init__()
        for cfg, keys in ((skip, SKIP_KEYS), (gap, GAP_KEYS), (fork, FORK_KEYS), (refine, POS_KEYS)):
            assert cfg is None or sorted(cfg) == sorted(keys), (cfg, keys)
        self.dq = dq
        # the soft layers of the output, no parameter, in this order; None = off
        self.skip = None if skip is None else dict(skip)                     # E399: the skip-link layer on the final wiring (`skip_suppress`)
        self.gap = None if gap is None else dict(gap)                        # E420: the gap-closing layer (`close_gaps`)
        self.fork = None if fork is None else dict(fork)                     # E423: the fork gate (`fork_gate`)
        self.refine = None if refine is None else dict(refine)               # E424: the position layer (`refine_positions`): the output then holds `pos`
        self.proj = nn.Linear(TOK, d)
        self.mlp = nn.Sequential(nn.Linear(NIN, d), nn.GELU(), nn.Linear(d, d))
        self.ln0 = nn.LayerNorm(d)
        self.layers = nn.ModuleList([SelfLayer(d, heads, ffn) for _ in range(layers)])
        self.tnorm, self.types = nn.LayerNorm(d), nn.Linear(d, NTYPE)       # ONE reading for both positions (after layer 1 and after the last layer)
        self.lnw = nn.LayerNorm(d)
        self.wq, self.wk = nn.Linear(d, 2 * dq), nn.Linear(d, 2 * dq)       # the two classes (short, long)
        self.wbias, self.dest = pair_mlp(2), nn.Linear(d, 2)                # bias^c(g_ij), the destination evidence b^c_j
        self.none_k, self.none_b = nn.Parameter(torch.randn(2, dq) * 0.1), nn.Parameter(torch.zeros(2))
        # the first Linear of (x_i, x_j, g_ij) split into three
        self.sc_i, self.sc_j, self.sc_g, self.sc_o = nn.Linear(d, 32), nn.Linear(d, 32, bias=False), nn.Linear(NG, 32, bias=False), nn.Linear(32, 2)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        # the Python mirror of `step` (no device sync per forward); kept by forward, set_step and load_state_dict
        self._step_py = 0
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
        """the wiring read of the module docstring on the features x (B, K, d): wb (B, 2, K, K) = bias^c(g_ij), logw (B, K) = the log key weight, cand the gate
        -> (the log-probabilities of `wiring_logp`, LayerNorm(x))"""
        B, K, _ = x.shape
        xn = self.lnw(x)
        q, k = (m(xn).view(B, K, 2, self.dq) for m in (self.wq, self.wk))
        z = torch.einsum("bicd,bjcd->bcij", q, k).float() / math.sqrt(self.dq) + wb + logw[:, None, None, :] + self.dest(xn).float().permute(0, 2, 1)[:, :, None, :]
        zn = torch.einsum("bicd,cd->bci", q, self.none_k).float() / math.sqrt(self.dq) + self.none_b[None, :, None]
        return wiring_logp(z, zn, cand), xn

    def forward(self, batch):
        b = {k: batch[k] if self.live_inputs else _const(batch[k]) for k in INPUTS}
        valid, pos, dirv = b["valid"].bool(), b["pos"].float(), b["dirv"].float()
        w_in = b["e"].float().sigmoid()
        g, allowed = JT.pair_geometry(pos, dirv, valid, e=w_in)
        lw = lambda w: (w + 1e-3).log()  # noqa: E731
        learned = self._step_py >= SWITCH
        chord = g[..., 2] * 20.0
        cand = torch.stack([allowed & (chord <= SHORT), allowed & (chord > SHORT)], 1)
        feat = torch.cat([pos / JT.HALF, fourier(pos), dirv, b["e"][..., None], b["e_raw"][..., None], b["flags"], b["s"][..., None], b["uout"][..., None],
                          b["uin"][..., None]], -1).float()
        x = self.ln0(self.proj(b["tok"].float()) + self.mlp(feat))
        x = self.layers[0](x, g, lw(w_in), valid)
        types1, w1 = self.read_types(x)
        for layer in self.layers[1:]:
            x = layer(x, g, lw(w1 if learned else w_in), valid)
        types3, w3 = self.read_types(x)
        wb = self.wbias(g).float().permute(0, 3, 1, 2)                              # bias^c(g_ij)
        logp, xn = self.read_wiring(x, wb, lw(w3 if learned else w_in), cand)          # the wiring layer
        pt = types3.softmax(-1)
        if self.skip is not None:                                           # E399
            logp = skip_suppress(logp, cand, pt[..., 0], chord, self.skip)
        if self.gap is not None:                                            # E420
            logp = close_gaps(logp, cand, pt[..., 0], chord, self.gap)
        if self.fork is not None:                                           # E423
            logp = fork_gate(logp, cand, pt, chord, self.fork)
        scale = 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[:, :, None] + self.sc_j(xn)[:, None] + self.sc_g(g))).float().sigmoid()
        if self.training:
            self.step += 1
            self._step_py += 1
        out = dict(logp=logp, cand=cand, scale=scale, types=types3, types1=types1, valid=valid, chord=chord)
        if self.refine is not None:                                         # E424: the delivered node positions
            out["pos"] = refine_positions(pos, dirv, valid, logp, cand, pt[..., 0], self.refine)
        return out


# ------------------------------------------------------------------------------------------------------------------------------ probabilities and decodes
def probs(out):
    """-> p (B, K, K) the probability of the link i -> j in the class of the pair (0 outside the gate), pnone (B, K, K) the none probability of the source i in
    the class of the pair, pt (B, K, 7) the type probabilities, keep (B, K) the kept nodes (valid and p_none < 0.5, the output rule), cand (B, K, K) the allowed
    gate, longm (B, K, K) chord above 8 m"""
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
    """the decode of record (output only): link i -> j when p^c(j|i) >= thr and p^c(j|i) >= p^c(none|i) and both ends are kept nodes (a fork source keeps both
    branches) -> keep (B, K), link / long (B, K, K) bool, p"""
    pr = probs(out)
    return _links(pr, (pr["p"] >= thr) & (pr["p"] >= pr["pnone"]))


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------------------------------------------------------------------ a synthetic batch
def synthetic_batch(B=8, K=48, seed=0):
    """random frames that follow the batch dict of the card (2a): per frame a lane of 10 tokens 2 m apart, a second lane 12 m behind its end (one long link), a
    third lane elsewhere, 4 duplicate tokens 0.5 m beside a lane token (they are in G(i) of its predecessor, assigned to no GT node), padding to K. Types and
    flags follow the token role, so a head can fit them"""
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
