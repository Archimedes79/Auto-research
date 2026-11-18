import os
p = os.path.join(os.environ["BFLC_ROOT"], "training", "data", "lane_live_tok.py")
s = open(p, encoding="utf-8").read()
a = "  A PASS proves parity at bs 4 only:"
assert s.count(a) == 1, s.count(a)
b = '''  RESULT (2026-11-17 / 18, GPU, 200 VAL frames, B' seeds 0 and 1, runs `E372.fixture_gpu_bs4` / `E372.fixture_gpu_bs8`, json `runs/e372_fixture/fixture_gpu_bs{4,8}.json`, code 0b7b6e4369 / 99d5f570de): **bs 4 PASS: the whole live path is exactly the cache on every frame of both seeds, every field zero, the same ordered K cells, the label counts equal (label loop 33 - 34 ms per batch of 4)**; bs 8 FAIL = ENV (the batch-size dependence of cuDNN, not a bug: the `labels` part is exact at bs 8): tok 1.56e-2 (two float16 ulps), pos 1.8e-3 / 2.0e-3 m, dirv 2.2e-2 / 6.5e-2, e up to 2.7 on one Matrix-NMS neighbour-order flip, valid differs in 5 slots, ytype 5 - 6, y 16 - 27 pair entries, label 6 - 13 per 200 frames (label loop 71 - 80 ms per batch of 8): the live read-out of the S4 card therefore runs B' at bs 4 (the head at bs 8 as the cache dump).
'''
s = s.replace(a, b + a)
open(p, "w", encoding="utf-8").write(s)
print("ok")
