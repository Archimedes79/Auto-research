"""E389 (A6, D127): the route-association head. A token-wise network over the cached B' tokens of a frame and ONE cross-attention from every token to the SD route tokens: which perceived token lies on the carriageway of the
navigation route (road) and on the lane the vehicle will drive (lane). The route is read at METRE scale (the map errors are 1 - 2 m): registration is the job.

    head = RouteHead(d=64, heads=4, geo_hidden=32, hidden=64, inputs="full", attn="ab")   # 33,738 parameters ("geo" = no `tok`, the geometry-only control: 21,130; "a" / "plain": below; all pinned in the selftest)
    out = head(batch)                                                  # dict(road (B, K), lane (B, K) logits float32, mass (B, K, heads) = the attention on REAL route keys, 0 for no map)
    loss, info = route_loss(out, batch, pos_weight=(wr, wl))           # BCE of both logits on the tokens with a label != -1
    batch = synthetic_batch(B, K, M, seed)                             # tests and the trainer's --smoke; check_batch(batch) = the data contract; label_counts(batch) -> the TRAIN frequencies of the labels

Batch = the `lane_ptok_data` batch of the cached B' tokens + batch["route"] = dict(pos (B, M, 2) m in the lane frame, dir (B, M, 2) unit direction of travel, count (B, M) int64 lane count 0 (unknown) - 5, valid (B, M) bool; padding
invalid) + batch["on_road"], batch["on_lane"] (B, K) int8 in {-1, 0, 1} (-1 = no label). The forward reads only INPUT_SETS[inputs] = tok (B, K, 64) ("full" only), pos, dirv (B, K, 2), e (B, K) existence logit, flags (B, K, 8) logits, and the route; NEVER a
label or any GT key of the batch, and the route has exactly the four keys (a fifth is a leak: refused).
Per token: x = [LN(Linear(tok)) ("full" only), MLP(pos / 50, clip(pos / 5, +-3), dirv, sigmoid(e), sigmoid(flags))] (64 + 32; the metre-scale own position lets a no-map head learn the straight-ahead corridor); q = Linear(LN(x)). Route key j:
LN(Linear([pos / 50, dir, one-hot(count)])) -> k, v (Linear). Pair features g_ij (`pair_features`, 8, METRE scale: lengths / 5 clipped to +-3): cos and sin of the turn from the token heading to the route direction, the offset route - token along / across the TOKEN
heading and along / across the ROUTE direction (the across offset of the key frame is the lateral offset of the token from the route line), the distance, the key's lane count / 5. Attention A (learned): per head logit_ij = q_i . k_j / sqrt(dh) + bias_h(g_ij);
the padding route keys are masked; ONE learned none key per head (k, v and a bias): the softmax runs over {the valid route keys} + {none}, a frame without a map (no valid key = arm z) attends to the none key only. Attention B (sharp, fixed, no parameter):
softmax over the valid keys of -distance / TAU (TAU = 0.3 m: the nearest route key takes ~all the weight, a soft argmin, no hard selection) -> the metre-scale features of the NEAREST key. feature = [x, the attended values of A (the none value included), the real-key
mass of A per head (the sum over the real keys: 0 without a map), g pooled by A (the head-mean attention), g pooled by B, `any` = 1 when the frame has a valid key]; an MLP (181 -> 64 -> 2, 117 for "geo") gives the two logits. No self-attention among the tokens: a
token's output depends on its own row and the route only.
`attn` (the architecture option; "ab" is the E389 head described above): "a" = ONE attention: A as it is, attention B and its pooled features are removed (the MLP reads 173 values, 109 for "geo"; 33,226 / 20,618 parameters); "plain" = "a" with the plain logit
q . k / sqrt(dh) (+ the none key) and NO pair-feature bias (no `bias` module; the pair features enter only as the pooled value and the mass; 32,806 / 20,198 parameters). Tests: grid/test_jroute.py
"""
import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TOK, D, NGEO, NCOUNT = 64, 64, 8, 6                                        # the B' node token, the width, the pair features, the lane counts 0 - 5
HALF, METRE, CLIP, TAU = 50.0, 5.0, 3.0, 0.3                               # the +- 50 m window; the metre scale of the features (length / METRE clipped to +-CLIP); the temperature of attention B in metres
NEG = -1e4                                                                 # the fill of a masked logit
NIN = 2 + 2 + 2 + 1 + 8                                                    # pos / HALF, clip(pos / METRE), dirv, p_exist, the 8 flag probabilities
INPUT_SETS = dict(full=("tok", "pos", "dirv", "e", "flags"), geo=("pos", "dirv", "e", "flags"))     # the batch keys the forward reads (besides the route) by the `inputs` option
ATTNS = ("ab", "a", "plain")                                               # the attention variants of the head (module docstring)
ROUTE_KEYS = ("pos", "dir", "count", "valid")                            # the fixed key set of the route tokens
LABELS = ("on_road", "on_lane")


def needed(inputs):
    """the batch keys the trainer moves to the device: nothing else of the batch (the GT keys, the dense pair targets) reaches the head"""
    return INPUT_SETS[inputs] + ("valid", "route") + LABELS


def pair_features(pos_q, dirv_q, pos_k, dirv_k, count_k):
    """pos / dirv (B, Kq, 2) and (B, Kk, 2), count_k (B, Kk) float -> (B, Kq, Kk, NGEO), [b, i, j] = the key j seen from the query i, lengths in metres / METRE clipped to +-CLIP: cos and sin of the turn from the query heading to the key heading (sin =
    cross(t_q, t_k)), the offset pos_k - pos_q along and across the query heading (across = its left normal (-t_y, t_x)), along and across the KEY heading, the distance (clipped to 0 .. CLIP), count_k / 5"""
    ch = pos_k[:, None, :, :] - pos_q[:, :, None, :]
    tq, tk = dirv_q[:, :, None, :].expand_as(ch), dirv_k[:, None, :, :].expand_as(ch)
    nq, nk = (torch.stack([-t[..., 1], t[..., 0]], -1) for t in (tq, tk))
    dot = lambda u, v: (u * v).sum(-1)  # noqa: E731
    m = lambda v: (v / METRE).clamp(-CLIP, CLIP)  # noqa: E731
    cross = tq[..., 0] * tk[..., 1] - tq[..., 1] * tk[..., 0]
    return torch.stack([dot(tq, tk), cross, m(dot(ch, tq)), m(dot(ch, nq)), m(dot(ch, tk)), m(dot(ch, nk)), ((dot(ch, ch) + 1e-12).sqrt() / METRE).clamp(max=CLIP), (count_k / (NCOUNT - 1))[:, None, :].expand_as(cross)], -1)


def check_route(route, B):
    """the shape contract of the route tokens (no value read, no device sync): exactly the four keys, (B, M, 2) / (B, M) shapes, bool valid, integer count -> M"""
    assert set(route) == set(ROUTE_KEYS), f"the route has the keys {sorted(route)}, it must have exactly {sorted(ROUTE_KEYS)} (any other key is a leak)"
    M = route["valid"].shape[1]
    assert route["pos"].shape == (B, M, 2) and route["dir"].shape == (B, M, 2) and route["count"].shape == (B, M) and route["valid"].shape == (B, M) and M >= 1, {k: tuple(v.shape) for k, v in route.items()}
    assert route["valid"].dtype == torch.bool and not route["count"].dtype.is_floating_point, (route["valid"].dtype, route["count"].dtype)
    return M


def own_features(pos, dirv, e, flags):
    """the geometry of the tokens themselves (B, K, NIN): pos / HALF, the metre-scale clip(pos / METRE), dirv, p_exist, the flag probabilities"""
    return torch.cat([pos / HALF, (pos / METRE).clamp(-CLIP, CLIP), dirv, e.sigmoid()[..., None], flags.sigmoid()], -1)


class RouteHead(nn.Module):
    """the module docstring; `attn` = "ab" (the E389 head) | "a" (one attention: no sharp attention B) | "plain" (a, and no pair-feature bias in the logit of A)"""

    def __init__(self, d=D, heads=4, geo_hidden=32, hidden=64, inputs="full", attn="ab"):
        super().__init__()
        assert d % heads == 0, f"d {d} is not a multiple of heads {heads}"
        assert inputs in INPUT_SETS, f"inputs {inputs!r}: {sorted(INPUT_SETS)}"
        assert attn in ATTNS, f"attn {attn!r}: {ATTNS}"
        self.h, self.dh, self.inputs, self.attn = heads, d // heads, inputs, attn
        nf = geo_hidden + (d if inputs == "full" else 0)
        if inputs == "full":
            self.tok = nn.Sequential(nn.Linear(TOK, d), nn.LayerNorm(d))                                      # LN(Linear(tok))
        self.geo = nn.Sequential(nn.Linear(NIN, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, geo_hidden))   # the token's own geometry and existence
        self.ln_q, self.q = nn.LayerNorm(nf), nn.Linear(nf, d)
        self.rkey = nn.Sequential(nn.Linear(4 + NCOUNT, d), nn.LayerNorm(d))                                  # the route key: [pos / HALF, dir, one-hot(count)]
        self.kv = nn.Linear(d, 2 * d)
        self.k_none, self.v_none = nn.Parameter(torch.randn(heads, self.dh) * 0.1), nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.none_b = nn.Parameter(torch.zeros(heads))
        if attn != "plain":
            self.bias = nn.Sequential(nn.Linear(NGEO, geo_hidden), nn.GELU(), nn.Linear(geo_hidden, heads))  # bias_h(g_ij)
        self.out = nn.Sequential(nn.Linear(nf + d + heads + (2 if attn == "ab" else 1) * NGEO + 1, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def parts(self, batch):
        """-> dict(x (B, K, nf), att (B, h, K, M + 1) float32 attention A = the softmax over {the route keys} + {none} (the none key LAST; a masked key has probability exactly 0), g (B, K, M, NGEO), att_b (B, K, M) float32 the sharp attention B
        (all 0 without a valid key), attended (B, K, d), mass (B, K, h) float32, pm_a, pm_b (B, K, NGEO) float32 the pair features pooled by A (head mean) / B, anyv (B, K, 1) float32): everything the output MLP reads; att_b and pm_b exist for attn = ab only"""
        b, rt = {k: batch[k] for k in INPUT_SETS[self.inputs]}, batch["route"]
        B, K = b["pos"].shape[:2]
        M = check_route(rt, B)
        pos, dirv = b["pos"].float(), b["dirv"].float()
        x = self.geo(own_features(pos, dirv, b["e"].float(), b["flags"].float()))
        if self.inputs == "full":
            x = torch.cat([self.tok(b["tok"].float()), x], -1)
        q = self.q(self.ln_q(x)).view(B, K, self.h, self.dh).transpose(1, 2)                                   # (B, h, K, dh)
        z_none = (q * self.k_none[None, :, None, :]).sum(-1).float() / math.sqrt(self.dh) + self.none_b[None, :, None]
        rpos, rdir, rvalid = rt["pos"].float(), rt["dir"].float(), rt["valid"].bool()
        rf = torch.cat([rpos / HALF, rdir, F.one_hot(rt["count"].long(), NCOUNT).float()], -1)
        k, v = self.kv(self.rkey(rf)).view(B, M, 2, self.h, self.dh).permute(2, 0, 3, 1, 4)                     # (B, h, M, dh) each
        g = pair_features(pos, dirv, rpos, rdir, rt["count"].float())
        z = (q @ k.transpose(-1, -2)).float() / math.sqrt(self.dh)
        if self.attn != "plain":
            z = z + self.bias(g).float().permute(0, 3, 1, 2)
        att = torch.cat([z.masked_fill(~rvalid[:, None, None, :], NEG), z_none[..., None]], -1).softmax(-1)
        vn = self.v_none[None, :, None, :].to(v.dtype)
        attended = (att[..., :-1].to(v.dtype) @ v + att[..., -1:].to(v.dtype) * vn).transpose(1, 2).reshape(B, K, -1)
        anyv = rvalid.any(1).float()[:, None, None]                                                            # (B, 1, 1)
        p = dict(x=x, att=att, g=g, attended=attended, mass=att[..., :-1].sum(-1).transpose(1, 2), pm_a=(att[..., :-1].mean(1)[..., None] * g.float()).sum(2), anyv=anyv.expand(B, K, 1))
        if self.attn == "ab":
            att_b = (-g[..., 6] * (METRE / TAU)).masked_fill(~rvalid[:, None, :], NEG).softmax(-1) * anyv       # the nearest key by distance, soft; all 0 without a key
            p.update(att_b=att_b, pm_b=(att_b[..., None] * g.float()).sum(2))
        return p

    def forward(self, batch):
        p = self.parts(batch)
        names = ("x", "attended", "mass", "pm_a") + (("pm_b",) if self.attn == "ab" else ()) + ("anyv",)
        lo = self.out(torch.cat([p[n].to(p["x"].dtype) for n in names], -1)).float()
        return dict(road=lo[..., 0], lane=lo[..., 1], mass=p["mass"])


# ------------------------------------------------------------------------------------------------------------------------------ the loss and the label frequencies
def route_loss(out, batch, pos_weight=(1.0, 1.0)):
    """BCE of the road and the lane logit on the tokens with a label != -1 (the label of a token that is assigned to no GT node, of a padding token and of a frame without a route is -1), a positive token weighted by pos_weight = (road, lane)
    (torch's `pos_weight`: the mean over the labelled tokens of w y (-log p) + (1 - y) (-log (1 - p))); loss = bce_road + bce_lane -> (loss, info: bce_road, bce_lane, acc_road, acc_lane (0 without a labelled token), n_road, n_lane = the labelled
    tokens, pos_road, pos_lane = the positives among them). A batch without a labelled token has the loss 0 with a graph (zero gradient)"""
    loss, info = out["road"].sum() * 0.0, {}
    for name, key, w in zip(("road", "lane"), LABELS, pos_weight):
        lo, y = out[name], batch[key].to(out[name].device)
        m = y >= 0
        n = m.sum()
        t = (y == 1).float()
        bce = F.binary_cross_entropy_with_logits(lo, t, reduction="none") * torch.where(y == 1, float(w), 1.0)
        term = (bce * m).sum() / n.clamp_min(1)
        loss = loss + term
        info.update({f"bce_{name}": float(term), f"acc_{name}": float((((lo > 0).float() == t) & m).sum() / n.clamp_min(1)), f"n_{name}": float(n), f"pos_{name}": float(((y == 1) & m).sum())})
    return loss, info


def label_counts(batch):
    """-> dict(road=(n_pos, n_neg), lane=(n_pos, n_neg)): the labelled tokens of a batch by class (the TRAIN frequencies of the trainer's positive weights)"""
    return {n: (int((batch[k] == 1).sum()), int((batch[k] == 0).sum())) for n, k in zip(("road", "lane"), LABELS)}


def pos_weights(counts):
    """counts = a list of `label_counts` -> (road, lane): n_neg / n_pos over all of them (the weight that balances the classes)"""
    pn = {n: np.sum([c[n] for c in counts], 0) for n in ("road", "lane")}
    assert all(p > 0 and q > 0 for p, q in pn.values()), f"a label class is empty: {pn}"
    return tuple(float(pn[n][1] / pn[n][0]) for n in ("road", "lane"))


def check_batch(batch):
    """the data contract (values: a device sync, so the trainer checks the first batch only): the route key set and shapes, count in 0 - 5, unit directions of the valid route tokens, the labels in {-1, 0, 1} and -1 on every padding token, a lane token is a road token"""
    B, K = batch["pos"].shape[:2]
    rt = batch["route"]
    check_route(rt, B)
    v = rt["valid"]
    assert int(rt["count"].min()) >= 0 and int(rt["count"].max()) <= NCOUNT - 1, "the lane count is outside 0 - 5"
    assert bool(((rt["dir"].norm(dim=-1) - 1).abs() <= 1e-3)[v].all()), "a valid route direction is not a unit vector"
    for key in LABELS:
        y = batch[key]
        assert y.shape == (B, K) and bool(((y >= -1) & (y <= 1)).all()), f"{key}: shape {tuple(y.shape)} or a value outside -1 / 0 / 1"
        assert bool((y[~batch["valid"].bool()] == -1).all()), f"{key}: a padding token has a label"
    assert bool((batch["on_road"][batch["on_lane"] == 1] == 1).all()), "a token on the ego lane is not on the road"


# ------------------------------------------------------------------------------------------------------------------------------ synthetic batches
def synthetic_batch(B=8, K=48, M=24, seed=0):
    """random frames in the layout of the batch of the module docstring: a curved route (2 m samples, 1 - 3 lanes, a random start and heading) per sample, half of the tokens placed along it (lateral noise 2.5 m, heading noise, 15 % against the direction of
    travel), the others anywhere; ~10 % of the tokens are padding. on_road = nearest route sample closer than count * 1.75 + 0.5 m with a heading cosine >= 0.7, on_lane = on_road and closer than 1.5 m, 15 % of the valid tokens have the label -1. The label depends
    on the route of the sample, so a head without the route cannot know it"""
    rng = np.random.RandomState(seed)
    th = rng.uniform(0, 2 * np.pi, (B, 1)) + rng.uniform(-0.03, 0.03, (B, 1)) * np.arange(M)[None]               # the heading along the route
    u = np.stack([-np.sin(th), np.cos(th)], -1)                                                                     # (B, M, 2)
    rpos = rng.uniform(-20, 20, (B, 1, 2)) + np.concatenate([np.zeros((B, 1, 2)), np.cumsum(2.0 * u[:, :-1], 1)], 1)
    rvalid = np.arange(M)[None] < rng.randint(max(M // 2, 1), M + 1, (B, 1))
    count = np.repeat(rng.randint(1, 4, (B, 1)), M, 1) * rvalid
    pos, valid = rng.uniform(-45, 45, (B, K, 2)), np.arange(K)[None] < K - rng.randint(0, max(K // 10, 1) + 1, (B, 1))
    ang = rng.uniform(0, 2 * np.pi, (B, K))
    near = np.arange(K)[None] < K // 2
    j = np.minimum(rng.randint(0, M, (B, K)), rvalid.sum(1)[:, None] - 1)                                           # a valid route sample each
    pj, uj = np.take_along_axis(rpos, j[..., None], 1), np.take_along_axis(u, j[..., None], 1)
    lat = np.stack([-uj[..., 1], uj[..., 0]], -1)
    pos = np.where(near[..., None], pj + lat * rng.randn(B, K, 1) * 2.5, pos)
    ang = np.where(near, np.arctan2(uj[..., 1], uj[..., 0]) + rng.randn(B, K) * 0.4 + np.where(rng.rand(B, K) < 0.15, np.pi, 0.0), ang)
    dirv = np.stack([np.cos(ang), np.sin(ang)], -1)
    dd = np.linalg.norm(pos[:, :, None] - rpos[:, None], axis=-1) + 1e9 * ~rvalid[:, None]                          # (B, K, M)
    n = dd.argmin(-1)
    dn, un, cn = np.take_along_axis(dd, n[..., None], 2)[..., 0], np.take_along_axis(u, n[..., None], 1), np.take_along_axis(count, n, 1)
    road = (dn < cn * 1.75 + 0.5) & ((dirv * un).sum(-1) >= 0.7) & valid
    lane = road & (dn < 1.5)
    return _pack(B, K, rng, valid, pos, dirv, road, lane, dict(pos=rpos, dir=u, count=count, valid=rvalid))


def straight_batch(B=8, K=48, seed=0):
    """the no-map task: tokens x in +-8 m across, y in +-45 m ahead, 60 % heading within 0.5 rad of straight ahead (+y), the rest anywhere; on_road = |x| <= 3.5 m and the heading cosine to +y >= 0.7 (the straight corridor prior), on_lane = on_road and
    |x| <= 1.75 m; no valid route key"""
    rng = np.random.RandomState(seed)
    valid = np.arange(K)[None] < K - rng.randint(0, max(K // 10, 1) + 1, (B, 1))
    pos = np.stack([rng.uniform(-8, 8, (B, K)), rng.uniform(-45, 45, (B, K))], -1)
    ang = np.where(rng.rand(B, K) < 0.6, rng.uniform(-0.5, 0.5, (B, K)), rng.uniform(-np.pi, np.pi, (B, K)))      # the angle from +y
    dirv = np.stack([np.sin(ang), np.cos(ang)], -1)
    road = (np.abs(pos[..., 0]) <= 3.5) & (dirv[..., 1] >= 0.7) & valid
    return _pack(B, K, rng, valid, pos, dirv, road, road & (np.abs(pos[..., 0]) <= 1.75), dict(pos=np.zeros((B, 1, 2)), dir=np.tile([0.0, 1.0], (B, 1, 1)), count=np.zeros((B, 1), np.int64), valid=np.zeros((B, 1), bool)))


def _pack(B, K, rng, valid, pos, dirv, road, lane, route):
    """the tensors of a synthetic batch: random token features, 15 % of the valid tokens unlabelled"""
    nolab = (rng.rand(B, K) < 0.15) | ~valid
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
    return dict(tok=t(rng.randn(B, K, TOK) * valid[..., None], torch.float16), pos=t(pos, torch.float32), dirv=t(dirv, torch.float32), e=t(np.where(valid, 2.0, -6.0) + 0.3 * rng.randn(B, K), torch.float32),
                flags=t(rng.randn(B, K, 8), torch.float32), valid=t(valid, torch.bool), route=dict(pos=t(route["pos"], torch.float32), dir=t(route["dir"], torch.float32), count=t(route["count"], torch.int64), valid=t(route["valid"], torch.bool)),
                on_road=t(np.where(nolab, -1, road), torch.int8), on_lane=t(np.where(nolab, -1, lane), torch.int8))


# ------------------------------------------------------------------------------------------------------------------------------ selftest (the mutants of test_jroute.py run it)
def _trained(seed=0, **kw):
    """a head with live attention at every part (the default start has a tiny geometry bias and uniform none bias)"""
    torch.manual_seed(seed)
    m = RouteHead(**kw)
    for p in ((m.bias[2].weight,) if m.attn != "plain" else ()) + (m.k_none, m.v_none):
        nn.init.normal_(p, std=0.5)
    nn.init.normal_(m.none_b, std=0.5)
    return m


def reference(m, batch):
    """the head by loops over batch, token, head and route key (plain float64 arithmetic of the module docstring) -> (road, lane, mass, pm_a, pm_b; pm_b is 0 unless attn = ab): the reference of the selftest"""
    P = {n: p.detach().double() for n, p in m.named_parameters()}
    ln = lambda t, n: (t - t.mean(-1, keepdim=True)) / (t.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt() * P[n + ".weight"] + P[n + ".bias"]  # noqa: E731
    lin = lambda t, n: t @ P[n + ".weight"].T + P[n + ".bias"]  # noqa: E731
    sc = lambda v: torch.clamp(v / METRE, -CLIP, CLIP)  # noqa: E731
    rt = batch["route"]
    B, K = batch["pos"].shape[:2]
    M = rt["valid"].shape[1]
    road, lane, mass, pm_a, pm_b = (torch.zeros(B, K, dtype=torch.float64), torch.zeros(B, K, dtype=torch.float64), torch.zeros(B, K, m.h, dtype=torch.float64), torch.zeros(B, K, NGEO, dtype=torch.float64),
                                    torch.zeros(B, K, NGEO, dtype=torch.float64))
    for bi in range(B):
        kv = [lin(ln(lin(torch.cat([rt["pos"][bi, j].double() / HALF, rt["dir"][bi, j].double(), F.one_hot(rt["count"][bi, j].long(), NCOUNT).double()]), "rkey.0"), "rkey.1"), "kv").view(2, m.h, m.dh) for j in range(M)]
        for i in range(K):
            pos, dirv, pe = batch["pos"][bi, i].double(), batch["dirv"][bi, i].double(), batch["e"][bi, i].double().sigmoid()
            x = lin(F.gelu(lin(torch.cat([pos / HALF, sc(pos), dirv, pe[None], batch["flags"][bi, i].double().sigmoid()]), "geo.0")), "geo.2")
            if m.inputs == "full":
                x = torch.cat([ln(lin(batch["tok"][bi, i].double(), "tok.0"), "tok.1"), x])
            q = lin(ln(x, "ln_q"), "q").view(m.h, m.dh)
            geos = {}
            for j in range(M):
                if not bool(rt["valid"][bi, j]):
                    continue
                tk, ch = rt["dir"][bi, j].double(), rt["pos"][bi, j].double() - pos
                nq, nk = torch.stack([-dirv[1], dirv[0]]), torch.stack([-tk[1], tk[0]])
                geos[j] = torch.stack([dirv @ tk, dirv[0] * tk[1] - dirv[1] * tk[0], sc(ch @ dirv), sc(ch @ nq), sc(ch @ tk), sc(ch @ nk), torch.clamp(ch.norm() / METRE, max=CLIP), rt["count"][bi, j].double() / (NCOUNT - 1)])
            att_out, a_mass, gsum = torch.zeros(m.h, m.dh, dtype=torch.float64), torch.zeros(m.h, dtype=torch.float64), torch.zeros(NGEO, dtype=torch.float64)
            for h in range(m.h):
                zs, vs = [], []
                for j, geo in geos.items():
                    zs.append(q[h] @ kv[j][0, h] / math.sqrt(m.dh) + (lin(F.gelu(lin(geo, "bias.0")), "bias.2")[h] if m.attn != "plain" else 0.0))
                    vs.append(kv[j][1, h])
                zs.append(q[h] @ P["k_none"][h] / math.sqrt(m.dh) + P["none_b"][h])
                vs.append(P["v_none"][h])
                a = torch.softmax(torch.stack(zs), 0)
                att_out[h] = (a[:, None] * torch.stack(vs)).sum(0)
                a_mass[h] = a[:-1].sum()
                if geos:
                    gsum = gsum + (a[:-1, None] * torch.stack(list(geos.values()))).sum(0) / m.h
            gb, anyv = torch.zeros(NGEO, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
            if geos:
                wb = torch.softmax(torch.stack([-g_[6] * METRE / TAU for g_ in geos.values()]), 0)
                gb, anyv = (wb[:, None] * torch.stack(list(geos.values()))).sum(0), torch.ones(1, dtype=torch.float64)
            if m.attn != "ab":
                gb = gb * 0.0
            o = lin(F.gelu(lin(torch.cat([x, att_out.reshape(-1), a_mass, gsum] + ([gb] if m.attn == "ab" else []) + [anyv]), "out.0")), "out.2")
            road[bi, i], lane[bi, i], mass[bi, i], pm_a[bi, i], pm_b[bi, i] = o[0], o[1], a_mass, gsum, gb
    return road, lane, mass, pm_a, pm_b


def _with_route(batch, **kw):
    return dict(batch, route=dict(batch["route"], **kw))


def _no_map(batch):
    return _with_route(batch, valid=torch.zeros_like(batch["route"]["valid"]))


def _fit(batch_fn, arm, steps, lr=4e-3, inputs="full", attn="ab"):
    """a short CPU fit of a fresh head on batch_fn(step); arm z = the route of every batch invalid -> the head"""
    torch.manual_seed(11)
    h = RouteHead(inputs=inputs, attn=attn)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        b = batch_fn(step)
        b = _no_map(b) if arm == "z" else b
        opt.zero_grad()
        route_loss(h(b), b)[0].backward()
        opt.step()
    return h


def _score(h, b, arm):
    """(accuracy, F1) of the road logit on the labelled tokens of b"""
    with torch.no_grad():
        lo = h(_no_map(b) if arm == "z" else b)["road"]
    lab = b["on_road"] >= 0
    pr, y = (lo > 0)[lab], (b["on_road"] == 1)[lab]
    return float((pr == y).float().mean()), 2 * int((pr & y).sum()) / max(int(pr.sum()) + int(y.sum()), 1)


def learn_tests(steps=300):
    """the learnability sanity tests: (1) a head WITH the route fits the exact-map rule (tokens on / off the carriageway band of a curved polyline, the label depends on the route of the sample) to > 0.95 accuracy, the same head without a
    map cannot; (2) a no-map head fits the straight corridor prior from the tokens' own position and heading (> 0.95); (3) the geometry-only control ("geo") fits the exact-map rule too; (4) the one-attention head (attn "a") does as (1)"""
    te = synthetic_batch(64, 32, 16, seed=9999)
    acc = {}
    for arm, attn in (("m", "ab"), ("z", "ab"), ("a_m", "a"), ("a_z", "a")):
        acc[arm] = _score(_fit(lambda s: synthetic_batch(16, 32, 16, seed=100 + s), arm[-1], steps, attn=attn), te, arm[-1])
    assert acc["m"][0] > 0.95 and acc["m"][1] > 0.9 and acc["z"][0] < 0.8 and acc["m"][1] - acc["z"][1] > 0.25, f"m reads the map, z cannot: (accuracy, F1) {acc}"
    assert acc["a_m"][0] > 0.95 and acc["a_m"][1] > 0.9 and acc["a_z"][0] < 0.8 and acc["a_m"][1] - acc["a_z"][1] > 0.25, f"attn a: m reads the map, z cannot: (accuracy, F1) {acc}"
    ts = straight_batch(64, 32, seed=9999)
    acc["straight"] = _score(_fit(lambda s: straight_batch(16, 32, seed=100 + s), "z", steps), ts, "z")
    assert acc["straight"][0] > 0.95 and acc["straight"][1] > 0.9, f"the no-map head cannot fit the straight corridor prior: {acc['straight']}"
    acc["geo"] = _score(_fit(lambda s: synthetic_batch(16, 32, 16, seed=100 + s), "m", steps, inputs="geo"), te, "m")
    assert acc["geo"][0] > 0.95 and acc["geo"][1] > 0.9, f"the geometry-only head cannot fit the exact-map rule: {acc['geo']}"
    return acc


def selftest():
    torch.manual_seed(0)
    B, K, M = 3, 9, 6
    batch = synthetic_batch(B, K, M, seed=1)
    check_batch(batch)
    # the size (a pinned number: a changed head is a new card): the full head and the geometry-only control
    sizes = {(i, a): sum(p.numel() for p in RouteHead(inputs=i, attn=a).parameters()) for i in ("full", "geo") for a in ATTNS}
    assert sizes == {("full", "ab"): 33738, ("geo", "ab"): 21130, ("full", "a"): 33226, ("geo", "a"): 20618, ("full", "plain"): 32806, ("geo", "plain"): 20198}, sizes
    heads = {a: RouteHead(attn=a) for a in ATTNS}
    assert heads["ab"].out[0].in_features == 181 and heads["a"].out[0].in_features == 173 and heads["plain"].out[0].in_features == 173 and RouteHead(inputs="geo", attn="a").out[0].in_features == 109, "the width of the output MLP"
    assert sorted({k.split(".")[0] for k in heads["ab"].state_dict()}) == sorted({k.split(".")[0] for k in heads["a"].state_dict()}) == ["bias", "geo", "k_none", "kv", "ln_q", "none_b", "out", "q", "rkey", "tok", "v_none"] and not hasattr(heads["plain"], "bias"), "the parameter blocks of the variants"
    try:
        RouteHead(attn="b")
        raise RuntimeError("an unknown attn is accepted")
    except AssertionError as e:
        assert "attn" in str(e), e
    # "a" is "ab" without the columns of the sharp pooled features: the same weights, those columns of out.0 cut -> the same output as "ab" with those columns zeroed; and it is another function than the "ab" head of the same seed
    mb = _trained(7)
    ma = _trained(7, attn="a")
    nb0 = mb.out[0].in_features - NGEO - 1                                                                      # the first column of pm_b: [x, attended, mass, pm_a | pm_b | anyv]
    sd = {k: v.clone() for k, v in mb.state_dict().items()}
    sd["out.0.weight"] = torch.cat([sd["out.0.weight"][:, :nb0], sd["out.0.weight"][:, nb0 + NGEO:]], 1)
    ma.load_state_dict(sd)
    mb.out[0].weight.data[:, nb0:nb0 + NGEO] = 0.0
    assert all(torch.allclose(ma(batch)[k], mb(batch)[k], atol=1e-5) for k in ("road", "lane", "mass")), '"a" is not "ab" without attention B'
    mb2 = _trained(7)
    assert not torch.allclose(mb2(batch)["road"], ma(batch)["road"], atol=1e-4) and "att_b" not in ma.parts(batch) and "pm_b" not in ma.parts(batch) and "att_b" in mb2.parts(batch), '"a" and "ab" are the same function'
    # pair_features by hand: the query at the origin heading north, the key at (3, 4) heading east with 2 lanes; a second query heading east; a far key (clipped)
    g = pair_features(torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]), torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]), torch.tensor([[[3.0, 4.0], [100.0, -80.0]]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), torch.tensor([[2.0, 5.0]]))
    # query 1 (north, n_q = (-1, 0)) vs key 1 (3, 4) east (n_k = (0, 1)): cos 0, sin -1, along t_q 4, across n_q -3, along t_k 3, across n_k 4, distance 5, 2 lanes; vs key 2 (100, -80) north: cos 1, sin 0, every length beyond 15 m: clipped to -3, distance 3, 5 lanes;
    # query 2 (east, n_q = (0, 1)) vs key 1: the same heading: cos 1, sin 0, along 3, across 4 (both frames); vs key 2: cos 0, sin 1, along t_q 100 -> 3, across n_q -80 -> -3, along t_k -80 -> -3, across n_k = (-1, 0) -100 -> -3
    want = torch.tensor([[[[0.0, -1.0, 4 / 5, -3 / 5, 3 / 5, 4 / 5, 1.0, 0.4], [1.0, 0.0, -3.0, -3.0, -3.0, -3.0, 3.0, 1.0]], [[1.0, 0.0, 3 / 5, 4 / 5, 3 / 5, 4 / 5, 1.0, 0.4], [0.0, 1.0, 3.0, -3.0, -3.0, -3.0, 3.0, 1.0]]]])
    assert g.shape == (1, 2, 2, NGEO) and torch.allclose(g, want, atol=1e-6), (g, want)
    # the loop reference of the whole head (live weights; a route with padding keys; samples without a map; padding tokens): the mixed batch and the all-invalid route (arm z); the geometry-only head too
    rv = batch["route"]["valid"].clone()
    rv[1] = False                                                                                              # sample 1 has no map
    for inputs, attn in itertools.product(("full", "geo"), ATTNS):
        m = _trained(inputs=inputs, attn=attn)
        for br in (batch, _with_route(batch, valid=rv), _no_map(batch)):
            out = m(br)
            road, lane, mass, pm_a, pm_b = reference(m, br)
            assert (out["road"].double() - road).abs().max() < 1e-4 and (out["lane"].double() - lane).abs().max() < 1e-4 and (out["mass"].double() - mass).abs().max() < 1e-5, (inputs, attn, "reference", float((out["road"].double() - road).abs().max()))
            pp = m.parts(br)
            assert (pp["pm_a"].double() - pm_a).abs().max() < 1e-5 and (attn != "ab" or (pp["pm_b"].double() - pm_b).abs().max() < 1e-5), (inputs, attn, "pooled features")
            assert out["road"].shape == (B, K) == out["lane"].shape and out["mass"].shape == (B, K, 4) and out["road"].dtype == torch.float32
            assert torch.allclose(pp["att"].sum(-1), torch.ones(B, 4, K), atol=1e-6) and torch.allclose(pp["mass"], 1 - pp["att"][..., -1].transpose(1, 2), atol=1e-6), "softmax over the keys + none; mass = 1 - the none attention"
    m = _trained()
    # the attention and the pooled features by hand: one token at the origin heading north (p_exist 0.5), two valid route keys + one padding key, every logit of A 0 (q = 0, the geometry bias 0, none_b 0): att = 1/3 on each real key and on none,
    # mass = 2/3 per head, pm_a = (g_1 + g_2) / 3 with g_1 = the key (3, 4) heading east, 2 lanes, g_2 = (0, 10) heading north, 0 lanes (by the 8 features of the module docstring); the sharp attention B takes key 1 (5 m against 10 m: weight 1 - 6e-8)
    hb = dict(tok=torch.randn(1, 1, TOK), pos=torch.zeros(1, 1, 2), dirv=torch.tensor([[[0.0, 1.0]]]), e=torch.zeros(1, 1), flags=torch.zeros(1, 1, 8), valid=torch.ones(1, 1, dtype=torch.bool),
              route=dict(pos=torch.tensor([[[3.0, 4.0], [0.0, 10.0], [0.0, 0.5]]]), dir=torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]), count=torch.tensor([[2, 0, 5]]), valid=torch.tensor([[True, True, False]])))      # the padding key sits next to the token
    g1 = torch.tensor([0.0, -1.0, 4 / 5, -3 / 5, 3 / 5, 4 / 5, 1.0, 0.4])
    g2 = torch.tensor([1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    for attn in ATTNS:                                                                                         # "plain" has no geometry bias to zero: its logits are 0 with q = 0 and none_b = 0 too
        mh = _trained(2, attn=attn)
        for p_ in (mh.q.weight, mh.q.bias, mh.none_b) + ((mh.bias[2].weight, mh.bias[2].bias) if attn != "plain" else ()):
            nn.init.zeros_(p_)
        ph = mh.parts(hb)
        assert torch.allclose(ph["att"][0, :, 0], torch.tensor([1 / 3, 1 / 3, 0.0, 1 / 3]).expand(4, 4), atol=1e-6) and torch.allclose(ph["mass"], torch.full((1, 1, 4), 2 / 3), atol=1e-6), (attn, ph["att"])
        assert torch.allclose(ph["pm_a"][0, 0], (g1 + g2) / 3, atol=1e-6) and torch.allclose(ph["g"][0, 0, 0], g1, atol=1e-6) and torch.allclose(ph["g"][0, 0, 1], g2, atol=1e-6), (attn, ph["pm_a"], (g1 + g2) / 3)
        assert ("att_b" in ph) == ("pm_b" in ph) == (attn == "ab") and float(ph["anyv"]) == 1.0, (attn, sorted(ph))
        if attn == "ab":
            assert torch.allclose(ph["pm_b"][0, 0], g1, atol=1e-5) and torch.allclose(ph["att_b"][0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5), (ph["pm_b"], ph["att_b"])
    # a token 1.0 m from a route polyline and one 6 m from it: the same token, the route 1 m / 6 m to its right (a line heading north, keys 10 m apart, 1 lane): the pooled features are the metre-scale offsets by hand (the nearest key (x, 0): across the heading 1 / 5 = 0.2
    # and 6 / 5 = 1.2, the distance the same, 1 lane 0.2; the other keys 10 m away or further take weight < 1e-8), and the two outputs differ
    ys = torch.arange(-20.0, 21.0, 10.0)
    def line(x):
        return dict(pos=torch.stack([torch.full_like(ys, x), ys], -1)[None], dir=torch.tensor([[[0.0, 1.0]] * 5]), count=torch.ones(1, 5, dtype=torch.int64), valid=torch.ones(1, 5, dtype=torch.bool))
    m_a, m_b = (_with_route(hb, **line(x)) for x in (1.0, 6.0))
    pa, pb = m.parts(m_a), m.parts(m_b)
    assert torch.allclose(pa["pm_b"][0, 0], torch.tensor([1.0, 0.0, 0.0, -0.2, 0.0, -0.2, 0.2, 0.2]), atol=1e-3) and torch.allclose(pb["pm_b"][0, 0], torch.tensor([1.0, 0.0, 0.0, -1.2, 0.0, -1.2, 1.2, 0.2]), atol=1e-3), (pa["pm_b"], pb["pm_b"])
    assert float((m(m_a)["road"] - m(m_b)["road"]).abs().max()) > 1e-4 and float((m(m_a)["lane"] - m(m_b)["lane"]).abs().max()) > 1e-4, "a token 1 m from the route and one 6 m from it get the same output"
    # the none key IS the no-map arm: the output with no valid route key is a function of the token alone (the none value, mass 0, pooled 0, any 0), the same for every route content, and equals the head on a route of ONE invalid key
    zr = torch.zeros_like(batch["route"]["valid"])
    junk = dict(pos=torch.full_like(batch["route"]["pos"], 77.0), dir=torch.randn_like(batch["route"]["dir"]), count=torch.full_like(batch["route"]["count"], 5), valid=zr)
    o_z, o_j = m(_no_map(batch)), m(_with_route(batch, **junk))
    assert all(torch.equal(o_z[k], o_j[k]) for k in o_z), "a padding route key changed the output"
    pz = m.parts(_no_map(batch))
    assert float(o_z["mass"].abs().max()) == 0.0 and all(float(pz[k].abs().max()) == 0.0 for k in ("pm_a", "pm_b", "att_b", "anyv")), "no map: no attention mass on a real key"
    one = dict(pos=batch["route"]["pos"][:, :1], dir=batch["route"]["dir"][:, :1], count=batch["route"]["count"][:, :1], valid=zr[:, :1])
    assert all(torch.allclose(o_z[k], m(dict(batch, route=one))[k], atol=1e-6) for k in o_z), "M does not matter without a valid key"
    # a sample without a map inside a batch is the all-invalid output; the others keep their map; the map changes the output
    o_m = m(_with_route(batch, valid=rv))
    full = m(batch)
    assert torch.allclose(o_m["road"][1], o_z["road"][1], atol=1e-6) and torch.allclose(o_m["road"][0], full["road"][0], atol=1e-6) and not torch.equal(full["road"][0], o_z["road"][0]) and float(o_m["mass"][0].min()) > 0, "a sample without a map reads the none key only"
    # route keys: a permutation leaves the output unchanged; tokens: a permutation of the tokens permutes the output (no interaction), a padding token's content changes no other token
    br = _with_route(batch, valid=rv)
    perm = torch.randperm(M)
    o_p = m(_with_route(batch, **{k: v[:, perm] for k, v in br["route"].items()}))
    assert torch.allclose(o_p["road"], m(br)["road"], atol=1e-5) and torch.allclose(o_p["lane"], m(br)["lane"], atol=1e-5), "not permutation invariant over the route keys"
    pt = torch.randperm(K)
    bp = {k: (v[:, pt] if k in INPUT_SETS["full"] + ("valid",) + LABELS else v) for k, v in batch.items()}
    assert torch.allclose(m(bp)["road"], full["road"][:, pt], atol=1e-5), "the tokens interact"
    pad = ~batch["valid"]
    bj = dict(batch, tok=batch["tok"].clone(), pos=batch["pos"].clone())
    bj["tok"][pad], bj["pos"][pad] = 9.0, 99.0
    assert torch.equal(m(bj)["road"][batch["valid"]], full["road"][batch["valid"]]), "a padding token changed a valid token"
    # the forward reads only INPUT_SETS[inputs] + the route (the geometry-only head never reads tok); a fifth key of the route is refused
    class Strict(dict):
        def __init__(self, d, allowed):
            super().__init__(d)
            self.allowed = allowed

        def __getitem__(self, k):
            assert k in self.allowed, f"the forward read the key {k!r}"
            return super().__getitem__(k)
    for inputs, keys in (("full", ("tok", "pos", "dirv", "e", "flags", "route")), ("geo", ("pos", "dirv", "e", "flags", "route"))):
        _trained(inputs=inputs)(Strict(batch, keys))
    leak = dict(batch, route=dict(batch["route"], piece=torch.zeros(B, M, dtype=torch.int64)))
    try:
        m(leak)
        raise RuntimeError("a fifth route key is accepted")
    except AssertionError as e:
        assert "leak" in str(e), e
    bl = dict(batch, on_road=1 - batch["on_road"].abs(), on_lane=torch.zeros_like(batch["on_lane"]))
    assert all(torch.equal(m(batch)[k], m(bl)[k]) for k in ("road", "lane")), "a label changed an output"
    # the loss by hand: logits (0, 2, -1, 3) on the road labels (1, 0, -1, 1) with the weight 3 for a positive: bce = (3 log 2 + (2 + log(1 + e^-2)) + 3 log(1 + e^-3)) / 3 on the 3 labelled tokens (the token with the label -1 and its logit 3 do not count)
    lo = torch.tensor([[0.0, 2.0, -1.0, 3.0]])
    yb = dict(on_road=torch.tensor([[1, 0, -1, 1]], dtype=torch.int8), on_lane=torch.tensor([[0, 0, 1, -1]], dtype=torch.int8))
    ls, inf = route_loss(dict(road=lo, lane=torch.tensor([[0.0, 0.0, 0.0, 0.0]])), yb, (3.0, 1.0))
    r_ = (3 * math.log(2) + (2 + math.log1p(math.exp(-2.0))) + 3 * math.log1p(math.exp(-3.0))) / 3
    l_ = 3 * math.log(2) / 3
    assert abs(inf["bce_road"] - r_) < 1e-5 and abs(inf["bce_lane"] - l_) < 1e-5 and abs(float(ls) - (r_ + l_)) < 1e-5, (inf, r_, l_)
    assert inf["n_road"] == 3 and inf["n_lane"] == 3 and inf["pos_road"] == 2 and inf["pos_lane"] == 1 and abs(inf["acc_road"] - 1 / 3) < 1e-6 and abs(inf["acc_lane"] - 2 / 3) < 1e-6, inf           # road: labels 1 / 0 / 1 vs logits 0 / 2 / 3 > 0: wrong, wrong, right; lane: labels 0 / 0 / 1 vs logits 0: right, right, wrong
    ls2, _ = route_loss(dict(road=torch.tensor([[0.0, 2.0, 55.0, 3.0]]), lane=torch.tensor([[0.0, 0.0, 0.0, -9.0]])), yb, (3.0, 1.0))
    assert abs(float(ls2) - float(ls)) < 1e-6, "a token with the label -1 counts"
    w13 = route_loss(dict(road=lo, lane=torch.zeros(1, 4)), yb, (1.0, 3.0))[1]
    assert abs(w13["bce_road"] - (math.log(2) + 2 + math.log1p(math.exp(-2.0)) + math.log1p(math.exp(-3.0))) / 3) < 1e-5 and abs(w13["bce_lane"] - (math.log(2) * 2 + 3 * math.log(2)) / 3) < 1e-5, "the weights are (road, lane)"
    nolab = dict(on_road=-torch.ones(1, 4, dtype=torch.int8), on_lane=-torch.ones(1, 4, dtype=torch.int8))
    lz = torch.zeros(1, 4, requires_grad=True)
    ln0, in0 = route_loss(dict(road=lz, lane=lz), nolab, (2.0, 2.0))
    ln0.backward()
    assert float(ln0) == 0.0 and in0["n_road"] == 0 and float(lz.grad.abs().sum()) == 0.0, "a batch without a label"
    # label frequencies: n_neg / n_pos over batches
    lc = label_counts(yb)
    assert lc == dict(road=(2, 1), lane=(1, 2)) and pos_weights([lc, dict(road=(1, 4), lane=(1, 1))]) == (5 / 3, 3 / 2), (lc, pos_weights([lc, dict(road=(1, 4), lane=(1, 1))]))
    # the gradient reaches EVERY parameter (a batch with maps, one sample without), for both input sets; with no map at all the route side gets none and the rest does
    big = synthetic_batch(4, 40, 12, seed=5)
    rv2 = big["route"]["valid"].clone()
    rv2[3] = False
    for inputs, attn in itertools.product(("full", "geo"), ATTNS):
        mg = _trained(3, inputs=inputs, attn=attn)
        route_loss(mg(_with_route(big, valid=rv2)), big)[0].backward()
        dead = [n for n, p in mg.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0]
        assert not dead, f"{inputs} {attn}: no gradient: {dead}"
        mz = _trained(3, inputs=inputs, attn=attn)
        route_loss(mz(_no_map(big)), big)[0].backward()
        live = {n.split(".")[0] for n, p in mz.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
        assert live == ({"tok", "geo", "out", "v_none"} if inputs == "full" else {"geo", "out", "v_none"}), f"{inputs} {attn}: arm z gradient (a softmax over the none key alone has no gradient into q, k_none, none_b): {sorted(live)}"
    # K = 512, M = 160, batch 8: finite, bf16 autocast finite, the evidence stays float32, memory below 2 GB on the CPU
    bb = _trained(4)
    xb = synthetic_batch(8, 512, 160, seed=6)
    ob = bb(xb)
    route_loss(ob, xb)[0].backward()
    assert ob["road"].shape == (8, 512) and torch.isfinite(ob["road"]).all() and all(torch.isfinite(p.grad).all() for p in bb.parameters())
    import resource
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 < 2048, "K = 512, M = 160, batch 8, forward and backward: more than 2 GB"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ab, pb_ = bb(xb), bb.parts(xb)
        assert torch.isfinite(ab["road"]).all() and torch.isfinite(ab["lane"]).all() and ab["road"].dtype == torch.float32 and all(pb_[k].dtype == torch.float32 for k in ("mass", "pm_a", "pm_b", "att_b", "att")), "bf16 autocast"
    # the data contract refuses the violations
    vm = batch["valid"]
    for what, bad in (("a lane token off the road", dict(on_road=torch.where(vm, 0, -1).to(torch.int8), on_lane=torch.where(vm, 1, -1).to(torch.int8))), ("a label on a padding token", dict(on_road=torch.where(vm, batch["on_road"], 1))),
                      ("a label outside -1 / 0 / 1", dict(on_road=batch["on_road"] + 5))):
        try:
            check_batch(dict(batch, **bad))
            raise RuntimeError(f"accepted: {what}")
        except AssertionError:
            pass
    for bad in (dict(count=batch["route"]["count"] + 9), dict(dir=batch["route"]["dir"] * 2.0)):
        try:
            check_batch(_with_route(batch, **bad))
            raise RuntimeError("accepted")
        except AssertionError:
            pass
    acc = learn_tests()
    print("jroute selftest ok (accuracy, F1):", {k: tuple(round(x, 3) for x in v) for k, v in acc.items()})


if __name__ == "__main__":
    import sys
    assert len(sys.argv) > 1 and sys.argv[1] in ("selftest", "learn"), "usage: jroute.py selftest | learn"
    selftest() if sys.argv[1] == "selftest" else print(learn_tests())
