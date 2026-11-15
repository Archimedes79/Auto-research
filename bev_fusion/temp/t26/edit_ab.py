p = "training/ab_timing.py"
s = open(p).read()
old1 = "            for _ in range(cfg.accum):\n                t1 = time.perf_counter()"
assert s.count(old1) == 1
s = s.replace(old1, "            for micro in range(cfg.accum):\n                t1 = time.perf_counter()")
old2 = "diag=False, backward=True, augment=True, scaler=self.scaler)"
assert s.count(old2) == 1
s = s.replace(old2, "diag=False, backward=True, augment=True, scaler=self.scaler, step=self.n, micro=micro)      # (step, micro: the image jitter's draw)")
open(p, "w").write(s)
