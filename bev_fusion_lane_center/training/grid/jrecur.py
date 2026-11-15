"""E354 (D103): recursive temporal fusion at feature level in the junction maps head: ONE ego-warped feature state of the previous keyframe, one input per step, the gradient through the state.

    w, valid = warp_state(s, R, t)                        # s (B, C, H, W) the state of the previous keyframe in ITS lane frame -> the same map in the lane frame of the current keyframe (bilinear, zero outside the old view), valid = the warp of ones
    jm = JMapsR(c2=144, c1=48, c05=32, d=64, ctx="dil+nopaint")
    out = jm(P, state=None, rel=None)                     # as `jmaps.JMaps` (j1, f1, [j05], [skip]) + out["state"] (B, d, 50, 50): the 2 m features AFTER the dilated blocks, the state for the next keyframe
    out = jm(P, state=prev["state"], rel=(R, t))          # rel = the pose of the previous lane frame in the current one, p_cur = R p_prev + t (`lane_seq_data.rel_pose`: R = Rc^T Rp, t = Rc^T (tp - tc)), R (B, 2, 2), t (B, 2)
    m = JRecurModel(maps_args, order=True, evidence=True, scales=True)     # the pair-stage model of `jends_train.JEndModel` with `JMapsR` as its maps head: same tensor names, a stored pair run loads with only `jmodel.jm.fuse.*` missing
    P, jout, out = m(x, gt=None, state=None, rel=None)    # as `JEndModel.forward`; the new state is jout["state"]
    m, args = load_model(run, dev)                        # a stored run of `jrecur_train.py`, or a pair run of `jends_train.py` from before E354 (no args['state']: the fusion stays at its zero gate)

`StateFuse`: c0' = c0 + g * conv3x3(cat[c0, w, valid]) between `p2` and the dilated blocks, g a per-channel gate initialised at ZERO: the start is the network without a state bit for bit. Without a state (the first keyframe of a scene, `--state off`)
w = 0 and valid = 0: the convolution still reads c0, so the equal-budget twin without a state has the same module. Everything is soft: the warp is a bilinear resampling by a known pose (differentiable in the features, the pose is data), no detach.
Axes as `jmaps.shift_along`: rows = x (right), columns = y (forward), extent +-50 m. Only the dilated context (`dil`, with or without `+nopaint` / `+nozone`), no `dirx`; the pair head without node tokens / attachment (the E341 recipe).
Tests: test_jrecur.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jends as JE  # noqa: E402
import jmaps as JM  # noqa: E402

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def warp_state(s, R, t, half=JM.HALF1):
    """s (B, C, H, W) in the lane frame of the PREVIOUS keyframe, read at p_prev = R^T (p_cur - t) for every cell centre p_cur of the current grid (R (B, 2, 2), t (B, 2): p_cur = R p_prev + t).
    -> (w (B, C, H, W), valid (B, 1, H, W) the same warp of ones: the share of the cell the old view covers), float32; differentiable in s"""
    B, _, H, W = s.shape
    dev = s.device
    ci = -half + (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) * (2 * half / H)
    cj = -half + (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) * (2 * half / W)
    p = torch.stack(torch.meshgrid(ci, cj, indexing="ij"), -1)                                    # (H, W, 2) the cell centres (x, y) of the current frame
    q = torch.einsum("bhwi,bij->bhwj", p[None] - t.float()[:, None, None, :], R.float())             # R^T (p - t) as a row vector: (p - t) R
    g = torch.stack([q[..., 1], q[..., 0]], -1) / half                                              # grid x = column = y, grid y = row = x (as `jmaps.shift_along`)
    w = F.grid_sample(s.float(), g, mode="bilinear", padding_mode="zeros", align_corners=False)
    valid = F.grid_sample(torch.ones(B, 1, H, W, device=dev), g, mode="bilinear", padding_mode="zeros", align_corners=False)
    return w, valid


class StateFuse(nn.Module):
    """c0' = c0 + g * conv3x3(cat[c0, w, valid]); g (1, d, 1, 1) starts at zero. state None = no past: w = 0, valid = 0. The warp and the fusion run in float32 also under autocast (the recurrent path does not collect bf16 error).
    `probe = True`: every forward WITH a state appends to `contrib` the share of the output that the state causes, RMS(fuse(c0, w, valid) - fuse(c0, 0, 0)) / RMS(c0) = RMS(g * conv(cat[0, w, valid])) / RMS(c0), under no_grad
    (the gate alone is no measure of the use of the past: it also scales the c0 part of the convolution and opens without a state)"""

    def __init__(self, d=64):
        super().__init__()
        self.conv = nn.Conv2d(2 * d + 1, d, 3, 1, 1, bias=False)
        self.g = nn.Parameter(torch.zeros(1, d, 1, 1))
        self.probe, self.contrib = False, []

    def forward(self, c0, state=None, rel=None):
        with torch.autocast(device_type=c0.device.type, enabled=False):
            c0 = c0.float()
            if state is None:
                w, valid = torch.zeros_like(c0), torch.zeros_like(c0[:, :1])
            else:
                w, valid = warp_state(state, *rel)
                if self.probe:
                    with torch.no_grad():
                        d = self.g * self.conv(torch.cat([torch.zeros_like(c0), w, valid], 1))
                        self.contrib.append(float(d.pow(2).mean().sqrt() / c0.pow(2).mean().sqrt().clamp_min(1e-8)))
            return c0 + self.g * self.conv(torch.cat([c0, w, valid], 1))


class JMapsR(JM.JMaps):
    """`jmaps.JMaps` with the state fusion between `p2` and the dilated blocks; forward = the forward of JMaps (ctx dil, no dirx) + the fusion + out["state"]"""

    def __init__(self, c2=144, c1=48, c05=32, d=64, ctx="dil", fuse=True):
        super().__init__(c2=c2, c1=c1, c05=c05, d=d, ctx=ctx)
        assert self.ctx_mode == "dil" and self.dirx is None, f"ctx {ctx}: the state is the output of the dilated blocks (dil, without dirx)"
        self.fuse = StateFuse(d) if fuse else None                               # created LAST: the random initialisation of every other tensor is that of JMaps (`JRecurModel` creates it after the pair head for the same reason)

    def forward(self, P, state=None, rel=None):
        up = lambda t, n: F.interpolate(t, scale_factor=n, mode="bilinear", align_corners=False)  # noqa: E731
        c = self.fuse(self.p2(P["feat2"]), state, rel)                           # (B, d, 50, 50) the 2 m features of the pyramid, with the warped state of the previous keyframe
        for b in self.blocks:
            c = b(c)
        x1 = self.m1(torch.cat([P["feat1"], up(c, 2)], 1))
        out = dict(j1=self.head1(x1), f1=x1, state=c.float())
        if self.paint:
            out["j05"] = self.head05(self.m05(torch.cat([P["feat05"], up(x1, 2)], 1)))
        if not self.zone:
            out["skip"] = ("zone",)
        return out


class JModelR(JM.JModel):
    """`jmaps.JModel` (B's pyramid + the junction maps head) with `JMapsR`; the pyramid stays under no_grad"""

    def __init__(self, width=48, f05=32, d=64, ctx="dil", fuse=True):
        nn.Module.__init__(self)
        from pyramid import PyramidHead
        self.width, self.ctx = width, ctx
        self.pyr = PyramidHead(width=width, f05=f05)
        self.jm = JMapsR(c2=3 * width, c1=width, c05=f05, d=d, ctx=ctx, fuse=fuse)

    def forward(self, x, state=None, rel=None):
        with torch.no_grad():
            P = self.pyr(x)
        return P, self.jm(P, state=state, rel=rel)


class JRecurModel(nn.Module):
    """the pair-stage model of `jends_train.JEndModel` (order / evidence / scales; no node tokens, no attachment) on `JModelR`. The tensors are created in the order of JEndModel and the fusion last, so a seed gives every
    other tensor the values it has in JEndModel. e2e (the pair loss reaches the maps head) is the only mode: the fusion lives in the maps head."""

    def __init__(self, maps_args, order=True, evidence=True, d=64, scales=False):
        super().__init__()
        ctx, dm = maps_args.get("ctx", "dil"), maps_args.get("d", 64)
        assert not (evidence and "nozone" in ctx), "evidence on a maps head without the zone / core supervision would read unsupervised channels (as `JEndModel`)"
        self.jmodel = JModelR(width=48, d=dm, ctx=ctx, fuse=False)
        self.je = JE.JEnds(d=d, order=order, evidence=evidence, f1=dm, scales=scales, paint_ev="nopaint" not in ctx)
        self.jmodel.jm.fuse = StateFuse(dm)

    def forward(self, x, gt=None, state=None, rel=None):
        P, out = self.jmodel(x, state=state, rel=rel)
        return P, out, self.je(out, P, gt=gt)

    def gate(self):
        """(d,) the absolute gate per channel"""
        return self.jmodel.jm.fuse.g.detach().abs().flatten()


def build(args):
    """the model of a stored run from its args (`jrecur_train.py` and `jends_train.py` store the same names)"""
    assert not any(args.get(k) for k in ("ntok", "unc", "attach")), "JRecurModel has no node tokens / position scale / attachment"
    return JRecurModel(args["maps_args"], order=not args["no_order"], evidence=not args["no_evidence"], scales=bool(args.get("scales", False)))


def load_model(run, dev):
    """-> (model in eval mode, args). args['state'] = on / off: a run of `jrecur_train.py` (every tensor must be there); without it the run is a pair run of `jends_train.py` from before E354: only the fusion is missing and stays at its zero gate"""
    ck = torch.load(os.path.join(ROOT, "runs", run, "model.pt"), map_location="cpu")
    m = build(ck["args"])
    miss, extra = m.load_state_dict(ck["state"], strict=False)
    assert not extra and (all(k.startswith("jmodel.jm.fuse.") for k in miss) if "state" not in ck["args"] else not miss), f"{run}: missing {miss[:5]}, unexpected {extra[:5]}"
    return m.to(dev).eval(), ck["args"]
