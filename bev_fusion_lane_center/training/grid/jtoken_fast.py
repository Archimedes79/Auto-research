"""Speed path of the E365 token head (2026-11-17, a measured profile: the step is GPU-bound, 60 % of it dense K x K pair work); the SAME function, parameters and state_dict as `jtoken.JTokenHead` / `jtoken.jtoken_loss`.

    head = FastJTokenHead(); head.load_state_dict(plain.state_dict())     # a plain head's weights run here and back (no extra parameter or buffer)
    out = head(batch)                                                      # training mode: the handle scales are NOT computed for all pairs (`out["scale"]` is None, `out["scale_at"](b, i, j)` gives them at chosen pairs);
                                                                           # eval mode: everything as `JTokenHead` (the read-outs of the cards read out["scale"])
    loss, info = fast_loss(out, batch, ...)                                # `jtoken.jtoken_loss` with no host sync: info values are device tensors (read them when you log), same loss and gradients

What changes against the plain head (nothing in the function):
  1. the handle-scale MLP (x_i, x_j, g_ij) -> 2 is evaluated at the pairs the loss reads (the long GT links with a fitted scale), not at all K x K pairs; same numbers at those pairs;
  2. the four pair MLPs 13 -> 32 -> out (three attention biases, the wiring bias) share their first Linear as ONE matmul over g and read g once (`pair_biases`); `compile_pairs()` wraps that stack in torch.compile;
  3. the loss has no `.any()`, boolean-mask indexing or `float(...)`: `_frame_mean` is a masked sum, the info values stay tensors, the scale term takes a fixed-shape pair list when the batch carries one
     (`sp_b`, `sp_i`, `sp_j`, `sp_y` (n, 2), `sp_ok`; the GPU loader makes it) and a `nonzero` of the dense mask otherwise (the one sync left, CPU-loader batches only).
Not covered (the plain head runs instead, `forward` delegates): live_inputs. Eval mode computes the dense handle scales as the plain head does, unless `scale_in_eval = False` (frozen teachers). Tests: test_jtoken_fast.py (parity with `JTokenHead` on a synthetic and, when the cache exists, a real batch).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jtoken as JK  # noqa: E402
import jtok as JT  # noqa: E402

NEG, SHORT, SWITCH, INPUTS = JK.NEG, JK.SHORT, JK.SWITCH, JK.INPUTS


def pair_biases(g, w1, b1, w2s, b2s, hid=32):
    """the pair MLPs 13 -> hid -> n_m on one g: w1 (M * hid, 13) / b1 (M * hid) the stacked first Linears, w2s / b2s the M second ones -> M tensors (B, n_m, K, K) float32 (the layout the attention adds)"""
    h = F.linear(g, w1, b1)
    outs = []
    for m, (w2, b2) in enumerate(zip(w2s, b2s)):
        o = F.linear(F.gelu(h[..., m * hid:(m + 1) * hid]), w2, b2)
        outs.append(o.float().permute(0, 3, 1, 2))
    return tuple(outs)


def _layer(layer, x, bias, logw, valid):
    """`jtoken.SelfLayer.forward` with the pair bias (B, H, K, K) given"""
    B, K, _ = x.shape
    q, k, v = layer.qkv(layer.ln1(x)).view(B, K, 3, layer.h, layer.dh).permute(2, 0, 3, 1, 4)
    a = (q @ k.transpose(-1, -2)).float() / math.sqrt(layer.dh) + bias + logw[:, None, None, :]
    a = a.masked_fill(~valid[:, None, None, :], NEG).softmax(-1)
    x = x + layer.o((a.to(v.dtype) @ v).transpose(1, 2).reshape(B, K, -1))
    return x + layer.ffn(layer.ln2(x))


class FastJTokenHead(JK.JTokenHead):
    """`jtoken.JTokenHead` with the speed path of the module docstring (plain head only; every other option runs the parent's forward)"""

    _pairs_fn = None
    scale_in_eval = True                                                    # False: eval mode skips the dense handle-scale MLP too (`out["scale"]` None, `out["scale_at"]` stays): for frozen teachers

    def compile_pairs(self, **kw):
        """wrap the pair-bias stack in torch.compile (fuses the GELUs and the casts; the first call compiles)"""
        self._pairs_fn = torch.compile(pair_biases, **kw)
        return self

    def _stack(self):
        mlps = [l.bias for l in self.layers] + [self.wbias]
        w1 = torch.cat([m[0].weight for m in mlps], 0)
        b1 = torch.cat([m[0].bias for m in mlps], 0)
        return w1, b1, [m[2].weight for m in mlps], [m[2].bias for m in mlps]

    def forward(self, batch):
        if self.live_inputs:
            return super().forward(batch)
        b = {k: JK._const(batch[k]) for k in INPUTS}
        valid, pos, dirv = b["valid"].bool(), b["pos"].float(), b["dirv"].float()
        w_in = b["e"].float().sigmoid()
        g, allowed = JT.pair_geometry(pos, dirv, valid, e=w_in)
        lw = lambda w: (w + 1e-3).log()  # noqa: E731
        learned = self._step_py >= SWITCH
        feat = torch.cat([pos / JT.HALF, JK.fourier(pos), dirv, b["e"][..., None], b["e_raw"][..., None], b["flags"], b["s"][..., None], b["uout"][..., None], b["uin"][..., None]], -1).float()
        w1, b1, w2s, b2s = self._stack()
        biases = (self._pairs_fn or pair_biases)(g, w1, b1, w2s, b2s)
        x = self.ln0(self.proj(b["tok"].float()) + self.mlp(feat))
        x = _layer(self.layers[0], x, biases[0], lw(w_in), valid)
        types1, w1k = self.read_types(x)
        for n, layer in enumerate(self.layers[1:], 2):
            x = _layer(layer, x, biases[n - 1], lw(w1k if learned else w_in), valid)
        types3, w3 = self.read_types(x)
        B, K, _ = x.shape
        xn = self.lnw(x)
        q, k = (m(xn).view(B, K, 2, self.dq) for m in (self.wq, self.wk))
        z = torch.einsum("bicd,bjcd->bcij", q, k).float() / math.sqrt(self.dq) + biases[-1] + lw(w3 if learned else w_in)[:, None, None, :] + self.dest(xn).float().permute(0, 2, 1)[:, :, None, :]
        zn = torch.einsum("bicd,cd->bci", q, self.none_k).float() / math.sqrt(self.dq) + self.none_b[None, :, None]
        chord = g[..., 2] * 20.0
        cand = torch.stack([allowed & (chord <= SHORT), allowed & (chord > SHORT)], 1)

        def scale_at(bi, ii, jj):
            """the handle scales (n, 2) of the pairs (bi, ii, jj): the MLP of (x_i, x_j, g_ij) of the plain head, at those pairs only"""
            return 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[bi, ii] + self.sc_j(xn)[bi, jj] + self.sc_g(g[bi, ii, jj]))).float().sigmoid()

        if self.training:
            self.step += 1
            self._step_py += 1
            scale = None
        elif self.scale_in_eval:
            scale = 0.05 + 1.2 * self.sc_o(F.gelu(self.sc_i(xn)[:, :, None] + self.sc_j(xn)[:, None] + self.sc_g(g))).float().sigmoid()
        else:
            scale = None                                                    # a frozen teacher (E379) does not read the scales: no dense scale MLP
        return dict(logp=JK.wiring_logp(z, zn, cand), cand=cand, scale=scale, scale_at=scale_at, types=types3, types1=types1, valid=valid, chord=chord)


def _source_rows(out):
    """(B, 2, K) bool: the sources of a class, the valid tokens with a candidate of it (the same as `jtoken.source_rows`)"""
    return out["cand"].any(-1) & out["valid"][:, None]


def _frame_mean_free(x, m, w=None):
    """`jtoken._frame_mean` without a host sync: per frame the mean of x over the entries of m, then the mean over the frames that have one (0 when none, with a zero gradient)"""
    per = (x * m).sum(1) / (m.sum(1).clamp_min(1) if w is None else (w * m).sum(1).clamp_min(1e-12))
    has = m.any(1).to(per.dtype)
    return (per * has).sum() / has.sum().clamp_min(1.0)


def fast_loss(out, batch, type_w=0.5, scale_w=0.5, none_weight=1.0, pos_loss="sum"):
    """`jtoken.jtoken_loss` (same arguments, same loss and gradients) without a host sync; info = device tensors (floats of the plain info under `float(...)`); see the module docstring for the scale term"""
    dev, cand, lp, valid = out["logp"].device, out["cand"], out["logp"], out["valid"]
    y, yl = batch["y"].to(dev) > 0, batch["ylong"].to(dev).bool()
    G = y[:, None] & cand
    yt = batch["ytype"].to(dev).long()
    nll = JK._source_nll(lp, G, none_weight, pos_loss)
    rows = _source_rows(out)
    nll_s, nll_l = (_frame_mean_free(nll[:, c], rows[:, c]) for c in range(2))
    wt = torch.where(yt == 0, none_weight, 1.0) if none_weight != 1.0 else None
    ce1, ce3 = (_frame_mean_free(F.cross_entropy(t.float().flatten(0, 1), yt.flatten(), reduction="none").view_as(yt) * (1.0 if wt is None else wt), valid, wt) for t in (out["types1"], out["types"]))
    loss = nll_s + nll_l + type_w * (ce1 + ce3)
    f32 = lambda t: t.detach().float()  # noqa: E731
    info = dict(nll_s=f32(nll_s), nll_l=f32(nll_l), ce1=f32(ce1), ce3=f32(ce3), gt_links=f32(y.sum()), pos_pairs=f32(G.sum()), gt_outside=f32((y & ~cand.any(1)).sum()),
                gt_class=f32((y & cand.any(1) & (yl != (out["chord"] > SHORT))).sum()))
    if not scale_w:
        return loss, info
    scale_at = out.get("scale_at")
    if "sp_b" in batch and scale_at is not None:                           # a fixed-shape pair list (the GPU loader): no mask indexing, no sync
        bi, ii, jj = batch["sp_b"].to(dev).long(), batch["sp_i"].to(dev).long(), batch["sp_j"].to(dev).long()
        ys = batch["sp_y"].to(dev).float()
        m = batch["sp_ok"].to(dev).bool() & cand[:, 1][bi, ii, jj] & torch.isfinite(ys[:, 0])
        ys = torch.where(m[:, None], ys, torch.zeros_like(ys))             # only the unselected rows are zeroed: a NaN in a selected row stays NaN, as in the plain loss
        d = ((scale_at(bi, ii, jj).float() - ys).abs() * m[:, None]).sum()
        l1 = d / (2.0 * m.sum().clamp_min(1))
        loss = loss + scale_w * l1
        info["scale_l1"] = f32(l1)
        return loss, info
    ys = batch["yscale"].to(dev).float()
    m = G[:, 1] & torch.isfinite(ys[..., 0])
    bi, ii, jj = m.nonzero(as_tuple=True)                                   # dynamic shape: one host sync (the dense CPU-loader batches)
    if bi.numel():
        sc = scale_at(bi, ii, jj).float() if out.get("scale") is None else out["scale"].float()[m]
        l1 = (sc - ys[m]).abs().mean()
        loss = loss + scale_w * l1
        info["scale_l1"] = f32(l1)
    return loss, info
