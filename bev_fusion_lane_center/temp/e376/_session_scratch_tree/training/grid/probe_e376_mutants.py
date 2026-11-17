"""throw-away: how far each textual mutant of test_jtoken.py lands from the hand value (deleted after use)"""
import test_jtoken as TJ
import jtoken as JK

out, b, frames = TJ._hand(TJ.FL)
out = TJ._with_logp(out)
z, zn = out["z"].detach().tolist(), out["zn"].detach().tolist()
per = [TJ._hand_softf(f, z[n], zn[n]) for n, f in enumerate(frames)]
term = sum(1.0 - f for f, ny in per if ny > 0) / sum(1 for _, ny in per if ny > 0)
pair_tp = float((out["logp"][:, 1, :, :6].exp() * ((b["y"] > 0)[:, None] & out["cand"])[:, 1]).sum())
print("hand term", term, "per", per)
muts = {"short class": [("_soft_f(lp[:, 1], G[:, 1], rows[:, 1])", "_soft_f(lp[:, 0], G[:, 0], rows[:, 0])")],
        "P none mass": [("pp = ((1.0 - p[..., K]) * rows).sum(-1)", "pp = (p[..., K] * rows).sum(-1)")],
        "Y pair": [("y = (G.any(-1) & rows).sum(-1).float()", "y = (G & rows[..., None]).sum((-2, -1)).float()")],
        "Y=0 frames": [("(1.0 - f[ny > 0]).mean()", "(1.0 - f).mean()")],
        "not added": [("loss = loss + soft_f_long * softf", "loss = loss + 0.0 * softf")],
        "TP pair (equivalent)": [("tp = ((p[..., :K] * G).sum(-1) * rows).sum(-1)", "tp = (p[..., :K] * G * rows[..., None]).sum((-2, -1))")],
        "P/Y swapped (equivalent)": [("2.0 * tp / (pp + y + 1e-6)", "2.0 * tp / (y + pp + 1e-6)")]}
l0 = float(JK.jtoken_loss(out, b)[0])
for n, e in muts.items():
    m = TJ._mutant(*e)
    l1, info = m.jtoken_loss(out, b, soft_f_long=0.7)
    print(f"{n:28s} softf_l {info['softf_l']:.6f} (hand {term:.6f}, diff {info['softf_l'] - term:+.6f}); loss diff / 0.7 {(float(l1) - l0) / 0.7:.6f}")
ob, bb, fb = TJ._hand()
ob = TJ._with_logp(ob)
for n, e in {"short gets W_long": [("[none_weight, none_weight_long]", "[none_weight_long, none_weight]")], "W_long ignored": [("    if none_weight_long is not None:\n", "    if False:\n")]}.items():
    m = TJ._mutant(*e)
    l, i = m.jtoken_loss(ob, bb, none_weight=0.25, none_weight_long=1.0)
    lr, ir = JK.jtoken_loss(ob, bb, none_weight=0.25, none_weight_long=1.0)
    print(f"{n:20s} loss {float(l):.6f} vs {float(lr):.6f}; nll_s {i['nll_s']:.6f} vs {ir['nll_s']:.6f}; nll_l {i['nll_l']:.6f} vs {ir['nll_l']:.6f}")
