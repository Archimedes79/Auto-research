"""adds static_weight (Doppler-derived static weight of a cell) to training/radar_encoder.py: function, module method, self-test block."""
p = "training/radar_encoder.py"
s = open(p, encoding="utf-8").read()

# 1 the function, before aux_weight
func = '''def static_weight(state, grid=None, K=K_SLOTS, v_max=rb.V_MAX, harmonics=rb.HARMONICS, sigma=1.0):
    """(B, 1, NX, NY) in [0, 1]: how static the detections of a cell look from their Doppler alone, no parameters.

    Per detection s = exp(-(vr / sigma)^2) with vr the ego-compensated radial velocity (a static reflector reads about 0); a cell takes the
    count-weighted mean over its K exact slots and the remainder slot (which stands for count - K detections, its vr is the decoded mean
    code). A cell without a detection is 1.0 (nothing to doubt; it is not written to a memory anyway). It is a NECESSARY condition only:
    the radial part of a crossing object is about 0 for one radar, so a weight of 1 does not prove 'static' - a learned velocity head can
    only lower it further (product), it cannot raise a Doppler-moving cell. Uses the Doppler of the stored raster, so it needs the
    adapter's compensated vr like everything else here."""
    grid = grid or Grid()
    u = unpack(state, grid, K, v_max, harmonics)
    valid = (u["valid"] > 0.5).to(state.dtype)
    n_rem = (state[:, 0] - K).clamp(min=0) * valid[:, K]                  # detections the remainder slot stands for
    wts = torch.cat([valid[:, :K], n_rem[:, None]], 1)                    # (B, K + 1, NX, NY)
    s = torch.exp(-(u["vr"] / sigma) ** 2)
    tot = wts.sum(1)
    w = (wts * s).sum(1) / tot.clamp(min=1.0)
    return torch.where(tot > 0, w, torch.ones_like(w))[:, None]


'''
anchor = "def aux_weight(step"
assert s.count(anchor) == 1
s = s.replace(anchor, func + anchor, 1)

# 2 the module method, before kept_fraction
meth = '''    def static_weight(self, state, sigma=1.0):
        """Doppler-derived static weight of every cell, (B, 1, X, Y) in [0, 1] (module function static_weight): a prior for a radar
        memory, to be multiplied with a learned static map; no parameters."""
        return static_weight(state, self.grid, self.K, self.v_max, self.harmonics, sigma)

'''
anchor = "    @torch.no_grad()\n    def kept_fraction("
assert s.count(anchor) == 1
s = s.replace(anchor, meth + anchor, 1)

# 3 self-test block 8
old = "    print(f\"radar_encoder selftest ok: exact round trip of eight fields"
assert s.count(old) == 1
block = '''    # 8 static weight from the Doppler: known cells (static, moving, mixed, overflow with a moving remainder, empty), two frames
    cases = [([0.0], [0.0]), ([8.0], [0.0]), ([0.0, 2.0], [0.0, 0.01]), ([0.0, 0.0, 0.0, 8.0, 8.0], [0.0, 0.01, 0.02, 0.3, 0.4])]
    parts = []
    for i, (vrs, dts) in enumerate(cases):
        dd = _synthetic(len(vrs), 1, grid, g, cells=torch.tensor([[20 + 4 * i, 30]] * len(vrs)))
        dd["vr"], dd["dt"], dd["b"] = torch.tensor(vrs), torch.tensor(dts), torch.zeros(len(vrs), dtype=torch.long)
        parts.append(dd)
    other = _synthetic(1, 1, grid, g, cells=torch.tensor([[20, 30]]))              # frame 1, the same cell, another content
    other["vr"], other["dt"], other["b"] = torch.tensor([6.0]), torch.tensor([0.0]), torch.ones(1, dtype=torch.long)
    parts.append(other)
    dd = {k: torch.cat([p_[k] for p_ in parts]) for k in parts[0]}
    sw = static_weight(rasterize_delta_slots(dd, 2, grid), grid)
    assert sw.shape == (2, 1, 120, 104) and torch.isfinite(sw).all() and float(sw.min()) >= 0 and float(sw.max()) <= 1
    exp_mixed = (1.0 + math.exp(-4.0)) / 2
    got = [float(sw[0, 0, 20 + 4 * i, 30]) for i in range(4)]
    assert abs(got[0] - 1.0) < 1e-4 and got[1] < 1e-6, got                         # static -> 1, moving -> 0
    assert abs(got[2] - exp_mixed) < 1e-3, (got[2], exp_mixed)                     # two detections: their mean
    assert abs(got[3] - 0.6) < 1e-3, got[3]                                        # 3 exact static + 2 in the moving remainder
    assert float(sw[0, 0, 0, 0]) == 1.0 and float(sw[1, 0, 20, 30]) < 1e-3, "empty cell 1.0, frames do not mix"
    assert torch.equal(m.static_weight(rasterize_delta_slots(dd, 2, grid)), sw), "module method = function"
'''
s = s.replace(old, block + old, 1)
s = s.replace("remainder, Doppler wrap, module + side loss gradients; side loss at the start",
              "remainder, Doppler wrap, module + side loss gradients, Doppler static weight (static 1, moving 0, mixed, overflow); side loss at the start", 1)
s = s.replace("`forward(..., with_state=True)` also returns the lossless raster (the explicit stream, bypass for heads that read raw values).\n",
              "`forward(..., with_state=True)` also returns the lossless raster (the explicit stream, bypass for heads that read raw values).\n"
              "`static_weight(state)` = a parameter-free static prior of every cell from the Doppler of the raster, for a radar memory.\n", 1)
open(p, "w", encoding="utf-8").write(s)
print("patched")
