import os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_loss2.py")
s = open(p, encoding="utf-8").read()
a = s.index("print(f\"{'loss family':52s}")
new = '''MULT = {"areas": 3, "lines": 3, "dense objects": 3, "object head": 3, "motion": 1, "camera aux": 1, "radar aux": 1}      # per optimiser micro-batch: sem + object head at 3 dt, motion and aux at dt 0
rows = []
print(f"{'loss family':52s} {'fwd ms':>8s} {'fwd+bwd':>8s} {'ops fwd+bwd':>12s} {'x dt':>5s}")
for name, f in fams:
    fwd = med(f)
    fb = med(lambda: f().backward())
    n = ops(lambda: f().backward())
    m = next((v for k, v in MULT.items() if name.startswith(k)), 0)         # the BCE-only variant row has no multiplier (not part of the sum)
    rows.append((name, fwd, fb, n, m))
    print(f"{name:52s} {fwd:8.1f} {fb:8.1f} {n:12d} {m:5d}")
tot_fb = sum(fb * m for _, _, fb, _, m in rows)
tot_f = sum(f * m for _, f, _, _, m in rows)
print(f"sum per micro-batch (x dt): forward {tot_f:.0f} ms, forward + backward {tot_fb:.0f} ms (CPU, 2 threads, load-dependent: ranking only)")
for key in ("camera aux", "radar aux"):
    r = next(r for r in rows if r[0].startswith(key))
    print(f"  {key}: {100 * r[1] * r[4] / tot_f:.1f} % of the forward, {100 * r[2] * r[4] / tot_fb:.1f} % of forward + backward")
aux = [r for r in rows if r[0].startswith(("camera aux", "radar aux"))]
print(f"  both aux heads: {100 * sum(r[1] * r[4] for r in aux) / tot_f:.1f} % of the forward, {100 * sum(r[2] * r[4] for r in aux) / tot_fb:.1f} % of forward + backward")
'''
open(p, "w", encoding="utf-8", newline="\n").write(s[:a] + new)
print("ok")
