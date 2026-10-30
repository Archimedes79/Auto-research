p = "tests/radar_codecs.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''R_TOL, PSI_TOL = 0.5, math.radians(3)''', '''R_TOL, PSI_TOL = 0.5, math.radians(3)
DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL = 0.5, 3.0, 0.1, 0.5''', 1)
s = s.replace('''dop = [k * math.pi * 0.5 / rb.V_MAX for k in H for _ in (0, 1)]
        self.tol = geo + dop + [3.0 / rb.RCS_SCALE, 0.1 / rb.AGE_SCALE, 0.5, 0.5]       # offsets: 0.5 cell = 0.25 m''',
 '''dop = [k * math.pi * DOP_TOL / rb.V_MAX for k in H for _ in (0, 1)]
        self.tol = geo + dop + [RCS_TOL / rb.RCS_SCALE, AGE_TOL / rb.AGE_SCALE, OFF_TOL, OFF_TOL]   # offsets: 0.5 cell = 0.25 m''', 1)
s = s.replace('''def level_e(steps, names=("abs", "delta")):''', '''def level_f(steps, scale=0.25):
    """tight tolerances (all x scale) and a second seed: does the codec still matter once the probe can pass the identity?"""
    global R_TOL, PSI_TOL, DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL
    R_TOL, PSI_TOL, DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL = (v * scale for v in (R_TOL, PSI_TOL, DOP_TOL, RCS_TOL, AGE_TOL, OFF_TOL))
    for k in rt.TOL:
        rt.TOL[k] *= scale
    print(f"F  tolerances x {scale}: " + ", ".join(f"{k} {v:.3f}" for k, v in rt.TOL.items()) + f"; probe leaky+id, C = 64, {steps} steps")
    tr, te = rt.load(set(range(8))), rt.load({8, 9})
    print("  codec / seed        | oracle   | held-out | train | gap  | held-out median: " + ", ".join(rt.TOL) + " | violating % per field")
    for name, seed in (("abs", 0), ("delta", 0), ("phys", 0), ("fourier", 0), ("abs", 1), ("delta", 1)):
        c = Codec(name)
        _, ok_or = eval_codec(c, None, te)
        net = train_codec(c, "leaky+id", 64, steps, tr, seed=seed)
        e, ok = eval_codec(c, net, te)
        _, ok_tr = eval_codec(c, net, tr)
        med = ", ".join(f"{np.median(e[k][torch.isfinite(e[k])].cpu().numpy()):.3f}" for k in rt.TOL)
        h, t = ok.float().mean().item(), ok_tr.float().mean().item()
        print(f"  {name + ' / seed ' + str(seed):19s} | {100 * ok_or.float().mean().item():7.1f}% | {100 * h:7.1f}% | {100 * t:4.1f}% | {100 * (t - h):+4.1f} | {med} | {fail_share(e)}", flush=True)


def level_e(steps, names=("abs", "delta")):''', 1)
s = s.replace('''    if "e" in sys.argv[2:]:
        level_e(steps)''', '''    if "f" in sys.argv[2:]:
        level_f(steps)
    elif "e" in sys.argv[2:]:
        level_e(steps)''', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok", s.count("OFF_TOL"))
