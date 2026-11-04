import os, sys, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
path = os.path.join(ROOT, "tests", "test_stream_temporal.py")
src = open(path, encoding="utf-8").read()
head = src[:src.index('print("A - C: stream = batch')]
g = {"__file__": path, "__name__": "dbg"}
exec(compile(head, path, "exec"), g)
globals().update(g)
r = build(cam_buffer="recurrent")
s = stream(r, [0, 1, 2]) + [None] * 3
s[3] = run(r, mk([3]), s[2]["state"], [(torch.full((B,), 1 / 6), VG, WG)])
s[4] = run(r, mk([4]), s[3]["state"], SEGS[3])
s[5] = run(r, mk([5]), s[4]["state"], SEGS[4])
k1 = run(r, mk([0, 1, 2]))
k2 = run(r, mk([3, 4, 5]), k1["state"], (torch.full((B,), 0.5), VG, WG))
fa, fb = flat(s[5]), flat(k2)
for k in fa:
    d = float((fa[k] - fb[k]).abs().max())
    if d > 1e-6:
        print(k, d)
print("k1 vs s2", gap(k1, s[2]))
used = sum((x[0] for x in SEGS[3] + SEGS[4]), torch.zeros(B))
s3 = run(r, mk([3]), s[2]["state"], [(torch.full((B,), 0.5) - used, VG, WG)])
s4 = run(r, mk([4]), s3["state"], SEGS[3]); s5 = run(r, mk([5]), s4["state"], SEGS[4])
print("same dt arithmetic:", gap(s5, k2))
fa, fb = flat(s5), flat(k2)
for k in fa:
    d = float((fa[k] - fb[k]).abs().max())
    if d > 1e-7:
        print(k, d)
