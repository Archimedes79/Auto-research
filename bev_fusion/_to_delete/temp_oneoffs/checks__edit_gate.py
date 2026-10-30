"""one-off: learned soft gate between the carried and the fresh velocity start (--gate) in tests/test_flow_explains.py:
recursive training on nuScenes keyframe triples, gated variant in the recursive and the sequence evaluation."""
p = "tests/test_flow_explains.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# 0 imports and flag
rep("from temporal import FlowPyramid, FlowRefine, TemporalPropagation, doppler_update, propagate  # noqa: E402",
    "from temporal import (FlowPyramid, FlowRefine, TemporalPropagation, doppler_update, landing, propagate,  # noqa: E402\n"
    "                      sample_at, steps_tensor)")
rep('''DATA = arg("--data", "synthetic")''', '''GATE = "--gate" in sys.argv                      # learned soft gate carried vs fresh start, recursive training (nuScenes)
if GATE:
    TAG = TAG + "_gate"
DATA = arg("--data", "synthetic")''')

# 1 the gate module
rep('''        self.dec = nn.Sequential(nn.Conv2d(C + 1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 3, 1))
''', '''        self.dec = nn.Sequential(nn.Conv2d(C + 1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 3, 1))
        if GATE:      # [features, fresh v, carried v, their residuals] -> softmax weights (fresh, carried); zero start = 0.5 / 0.5
            self.gate = nn.Sequential(nn.Conv2d(C + 6, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, 2, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        self.last_gate = None

    @staticmethod
    def residual(f0, f1, steps, v):
        """(B, 1, X, Y): how badly a velocity explains the next frame at each SOURCE cell - |F0(p) - F1(landing(p))| summed
        over the channels (raw maps, bilinear lookup of F1 at the landing point)."""
        S = steps_tensor(steps, f0.shape[0], f0.device, f0.dtype)
        look = sample_at(f1, landing(S, v, G), G).view_as(f0)
        return (f0 - look).abs().sum(1, keepdim=True)
''')

# 2 forward: optional carried estimate, blended with the fresh start by the gate
rep('''    def forward(self, f0, f1, steps, v_override=None, vgt=None, v_init=None):''',
    '''    def forward(self, f0, f1, steps, v_override=None, vgt=None, v_init=None, v_carried=None):''')
rep('''            meas = radar_measure(f0, vgt)
            v0 = radar_apply(v0, meas)
            corr = lambda v: radar_apply(v, meas)                     # noqa: E731
''', '''            meas = radar_measure(f0, vgt)
            v0 = radar_apply(v0, meas)
            corr = lambda v: radar_apply(v, meas)                     # noqa: E731
            if v_carried is not None:
                v_carried = radar_apply(v_carried, meas)
        if v_carried is not None and GATE and v_override is None:     # learned soft blend of the two starts (no hard choice)
            rf_, rc_ = self.residual(f0, f1, steps, v0), self.residual(f0, f1, steps, v_carried)
            w = torch.softmax(self.gate(torch.cat([e0, v0 / 10.0, v_carried / 10.0, rf_, rc_], 1)), 1)
            self.last_gate = w.detach()
            v0 = w[:, :1] * v0 + w[:, 1:] * v_carried
''')

# 3 one loss function for a pair (used by the pair training and both pairs of the recursive training)
rep('''def train(model, data, iters, rng, bs=8, lr=2e-3):''', '''def pair_loss(model, src, tgt, steps, flows, preds, mix):
    """the training loss of one frame pair (no velocity labels): prediction BCE per refinement output, entropy,
    smoothness, raw warp, zero-motion prior - the same terms as the pair training."""
    loss = F.binary_cross_entropy_with_logits(mix, tgt)
    for k, p in enumerate(preds[1:]):
        loss = loss + 0.8 ** (len(preds) - 2 - k) * F.binary_cross_entropy_with_logits(p, tgt)
    if SHARP:
        loss = loss + ENT * (src.max(1).values * model.ent).mean()
    if SMOOTH > 0:
        loss = loss + SMOOTH * sum(smoothness(v, src) for v in flows)
    if RAWWARP > 0:
        loss = loss + RAWWARP * sum(((propagate(src, steps, v, G)[0] - tgt) ** 2).mean() for v in flows[1:])
    if PRIOR > 0:
        occ = src.max(1).values
        loss = loss + PRIOR * sum((occ * (v / 10.0).pow(2).sum(1)).mean() for v in flows)
    return loss


def train(model, data, iters, rng, bs=8, lr=2e-3):''')
rep('''        flows, preds, mix = model(m[:, 0], m[:, 1], steps, vgt=vg)
        tgt = m[:, 1]
        loss = F.binary_cross_entropy_with_logits(mix, tgt)
        for k, p in enumerate(preds[1:]):
            loss = loss + 0.8 ** (len(preds) - 2 - k) * F.binary_cross_entropy_with_logits(p, tgt)
        if SHARP:
            loss = loss + ENT * (m[:, 0].max(1).values * model.ent).mean()
        if SMOOTH > 0:
            loss = loss + SMOOTH * sum(smoothness(v, m[:, 0]) for v in flows)
        if RAWWARP > 0:                       # the explanation as a training term: v alone must move the raw map onto F1
            loss = loss + RAWWARP * sum(((propagate(m[:, 0], steps, v, G)[0] - tgt) ** 2).mean() for v in flows[1:])
        if PRIOR > 0:
            occ = m[:, 0].max(1).values
            loss = loss + PRIOR * sum((occ * (v / 10.0).pow(2).sum(1)).mean() for v in flows)
''', '''        flows, preds, mix = model(m[:, 0], m[:, 1], steps, vgt=vg)
        loss = pair_loss(model, m[:, 0], m[:, 1], steps, flows, preds, mix)
        if GATE:          # recursive: the estimate of pair 1 carried to F1 (detached) is blended with pair 2's fresh start
            s12 = [b[4] for b in batch]
            vg1 = torch.stack([b[5] for b in batch]).to(DEV)
            carried = carry(flows[-1].detach(), steps)
            f2, p2, mix2 = model(m[:, 1], m[:, 2], s12, vgt=vg1, v_carried=carried)
            loss = loss + pair_loss(model, m[:, 1], m[:, 2], s12, f2, p2, mix2)
''')

# 4 the recursive evaluation: + gated
rep('''        recur = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV), v_init=carried)[0][-1]''',
    '''        recur = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV), v_init=carried)[0][-1]
        gated = model(m[:, 1], m[:, 2], [s12], vgt=vgt1[None].to(DEV), v_carried=carried)[0][-1] if GATE else None''')
rep('''        for name, v in (("fresh", fresh), ("carried", carried), ("recursive", recur)):
            e = (v[0].cpu() - vgt1).norm(dim=0)''', '''        for name, v in (("fresh", fresh), ("carried", carried), ("recursive", recur), ("gated", gated)):
            if v is None:
                continue
            e = (v[0].cpu() - vgt1).norm(dim=0)''')

# 5 the sequence report: + a gated chain (its own carried estimate)
rep('''    acc = {(n, k, t): [0.0, 0] for n in ("fresh", "recursive") for k in ("vehicles", "moving") for t in range(T)}''',
    '''    names = ("fresh", "recursive") + (("gated",) if GATE else ())
    acc = {(n, k, t): [0.0, 0] for n in names for k in ("vehicles", "moving") for t in range(T)}''')
rep('''        carried = None
        for t in range(T):
            vg = vel[t][None].to(DEV)
            fresh = model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg)[0][-1]
            recur = fresh if carried is None else model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg, v_init=carried)[0][-1]
            carried = carry(recur, [steps[t]])''', '''        carried, carried_g = None, None
        for t in range(T):
            vg = vel[t][None].to(DEV)
            fresh = model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg)[0][-1]
            recur = fresh if carried is None else model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg, v_init=carried)[0][-1]
            carried = carry(recur, [steps[t]])
            gated = None
            if GATE:
                gated = fresh if carried_g is None else model(m[:, t], m[:, t + 1], [steps[t]], vgt=vg, v_carried=carried_g)[0][-1]
                carried_g = carry(gated, [steps[t]])''')
rep('''                for n, v in (("fresh", fresh), ("recursive", recur)):
                    e = (v[0].cpu() - vel[t]).norm(dim=0)''', '''                for n, v in (("fresh", fresh), ("recursive", recur), ("gated", gated)):
                    if v is None:
                        continue
                    e = (v[0].cpu() - vel[t]).norm(dim=0)''')
rep('''    print("  t   vehicles fresh / recursive    moving vehicles fresh / recursive")
    for t in range(T):
        print(f"  {t}   {res[('fresh', 'vehicles', t)]:6.2f} / {res[('recursive', 'vehicles', t)]:6.2f}"
              f"            {res[('fresh', 'moving', t)]:6.2f} / {res[('recursive', 'moving', t)]:6.2f}", flush=True)''',
    '''    print("  t   vehicles " + " / ".join(names) + "    moving vehicles " + " / ".join(names))
    for t in range(T):
        print(f"  {t}   " + " / ".join(f"{res[(n, 'vehicles', t)]:6.2f}" for n in names) + "            "
              + " / ".join(f"{res[(n, 'moving', t)]:6.2f}" for n in names), flush=True)''')
rep('''        for n, sty in (("fresh", "o--"), ("recursive", "s-")):''',
    '''        for n, sty in (("fresh", "o--"), ("recursive", "s-"), ("gated", "^-"))[:len(names)]:''')

# 6 main: --gate needs the triples with the F1 velocity
rep('''        rec = "--recursive" in sys.argv''', '''        rec = "--recursive" in sys.argv or GATE''')
rep('''    if DATA == "nuscenes" and "--recursive" in sys.argv:''', '''    if DATA == "nuscenes" and ("--recursive" in sys.argv or GATE):''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
