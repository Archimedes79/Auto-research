"""E299 (D90): the recursive state of the junction pair head: ONE ego-warped OUTPUT state of the previous keyframe, read by a gated cross-attention (`JEnds(state=True)`, flag off = E312 / E320 / E324 bit for bit).

    build_state(out, has=None) -> state       the OUTPUT of `JEnds` of a frame (keypoints, headings, probabilities, the pair logits), DETACHED: a constant input, no gradient through time
    warp_state(state, R, t) -> state          the entries in the lane frame of the NEXT frame: p' = R p + t, d' = R d (`R`, `t` = `lane_gt_data.rel_pose`)
    StateAttn(d=64)(t, pos, D, state) -> t    the keypoint tokens + gate x (cross-attention to the warped entries); a frame without a previous frame (state['has'] False) keeps its tokens

The state is the delivered output of the previous frame (not hidden tokens: they live in another embedding space), so the same network can be run on a stream at inference (the previous output is the buffer) and trained with
its own no-grad forward on the previous row. Entries 0 .. K-1 are the entry keypoints, K .. 2K-1 the exits (as in `JEnds`). An entry's state features: warped position, heading, probability, kind and a SUMMARY of its row
of pair logits (maximum, its probability, the expected partner by softmax over the allowed partners relative to the entry, the partner's heading): a summary, because the slot order of the previous frame means nothing now.
Hard steps (D55, named): none new inside the layer (soft association by the warp: a Gaussian prior of 3 m / 30 degrees + the content score, a learned 'no counterpart' slot); the state itself is a constant (detached).
"""
import math

import torch
import torch.nn as nn

HALF = 50.0                                                  # the 100 m BEV: positions are scaled by it
TOL_M, TOL_DEG = 3.0, 30.0                                   # the association tolerance across two keyframes (the card's step 0: `evaluation/jtime.py`)
NF = 12                                                      # features per state entry


def warp(pos, dirv, R, t):
    """pos, dirv (B, n, 2) in the lane frame of the previous frame -> the lane frame of the current one; R (B, 2, 2), t (B, 2): p_cur = R p_prev + t, d_cur = R d_prev"""
    Rt = R.transpose(1, 2)
    return pos @ Rt + t[:, None, :], dirv @ Rt


def build_state(out, has=None):
    """the state dict of a `JEnds` output: pos, dir, p (B, 2K, .), valid (B, 2K), logit / allowed (B, K, K), has (B,) bool (False = this row has no previous frame); every tensor DETACHED"""
    B = out["kp_pos"].shape[0]
    st = dict(pos=out["kp_pos"], dir=out["kp_dir"], p=out["kp_p"], valid=out["valid"].flatten(1), logit=out["logit"].float(), allowed=out["allowed"],
              has=torch.ones(B, dtype=torch.bool, device=out["kp_pos"].device) if has is None else has)
    return {k: v.detach() for k, v in st.items()}


def warp_state(state, R, t):
    pos, d = warp(state["pos"], state["dir"], R, t)
    return dict(state, pos=pos, dir=d)


def _row_summary(z, allowed, pos, dirv, own):
    """z, allowed (B, n, m): the pair logits of n entries (rows) towards m partners of the other kind; pos, dirv (B, m, 2) the partners, own (B, n, 2) the entries -> (B, n, 6): maximum logit / 8, its probability,
    the expected partner (softmax over the allowed ones) relative to the entry / 20 (2) and its heading (2); zeros for an entry without an allowed partner"""
    any_ = allowed.any(-1)
    zm = z.masked_fill(~allowed, -1e4)
    w = zm.softmax(-1) * any_[..., None]
    mx = torch.where(any_, zm.max(-1)[0].clamp(-8.0, 8.0), torch.full_like(zm[..., 0], -8.0))
    return torch.cat([(mx / 8.0)[..., None], (mx.sigmoid() * any_)[..., None], (w @ pos - own) / 20.0 * any_[..., None], (w @ dirv) * any_[..., None]], -1)


def state_features(state):
    """(B, 2K, NF) features of the (warped) state entries"""
    pos, d, p, z, al = state["pos"], state["dir"], state["p"], state["logit"], state["allowed"]
    K = z.shape[1]
    kind = torch.cat([torch.zeros(K), torch.ones(K)]).to(pos.device)
    ent = _row_summary(z, al, pos[:, K:], d[:, K:], pos[:, :K])                          # an entry's partners are the exits
    ext = _row_summary(z.transpose(1, 2), al.transpose(1, 2), pos[:, :K], d[:, :K], pos[:, K:])
    return torch.cat([pos / HALF, d, p[..., None], kind.expand(pos.shape[0], -1)[..., None], torch.cat([ent, ext], 1)], -1)


class StateAttn(nn.Module):
    """token_i <- token_i + gate x MLP([token_i, the state features averaged with the attention weights, the weighted offset of the matched entries / 3 m, the matched mass]); the attention of keypoint i over the
    state entries a of its kind: content (q . k / sqrt(dh)) - 0.5 (distance / 3 m)^2 - 0.5 (1 - cos heading) / (1 - cos 30 degrees), a learned null slot takes the mass of a keypoint with no counterpart (its matched
    mass is then small). The gate starts at 0 (a scalar): the model is the flag-off model at the start; the layer weights get a gradient only once the gate has moved (the gate itself has one at 0)."""

    def __init__(self, d=64, dh=32, hid=64):
        super().__init__()
        self.dh = dh
        self.q, self.k = nn.Linear(d, dh), nn.Linear(NF, dh)
        self.mlp = nn.Sequential(nn.Linear(d + NF + 3, hid), nn.GELU(), nn.Linear(hid, d))
        self.null = nn.Parameter(torch.zeros(1))
        self.gate = nn.Parameter(torch.zeros(1))

    def attend(self, t, pos, D, state, sf):
        """(B, n, n) attention weights of every keypoint over the state entries of its kind (valid entries only); the rest of the mass is the null slot's"""
        B, n = t.shape[0], t.shape[1]
        sp, sd = state["pos"], state["dir"]
        kind = torch.arange(n, device=t.device) >= n // 2
        d2 = ((pos[:, :, None, :] - sp[:, None, :, :]) ** 2).sum(-1)
        cos = (D[:, :, None, :] * sd[:, None, :, :]).sum(-1)
        s = self.q(t).float() @ self.k(sf).float().transpose(1, 2) / math.sqrt(self.dh) - 0.5 * d2 / TOL_M ** 2 - 0.5 * (1.0 - cos) / (1.0 - math.cos(math.radians(TOL_DEG)))
        s = s.masked_fill(~((kind[:, None] == kind[None, :])[None] & state["valid"][:, None, :]), float("-inf"))
        return torch.cat([s, self.null.expand(B, n, 1)], -1).softmax(-1)[..., :-1]

    def forward(self, t, pos, D, state):
        sf = state_features(state)
        w = self.attend(t, pos, D, state, sf)
        off = (w[..., None] * (state["pos"][:, None, :, :] - pos[:, :, None, :])).sum(2) / TOL_M
        upd = self.mlp(torch.cat([t, (w @ sf).to(t.dtype), off.to(t.dtype), w.sum(-1, keepdim=True).to(t.dtype)], -1))
        return torch.where(state["has"][:, None, None], t + (self.gate * upd).to(t.dtype), t)
