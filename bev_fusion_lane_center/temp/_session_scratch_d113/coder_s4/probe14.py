import os, numpy as np
R = os.environ["BFLC_ROOT"]
d = os.path.join(R, "runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val")
t = np.load(os.path.join(d, "tok.npy"), mmap_mode="r")[:40].astype(np.float32)
a = np.abs(t)
print("max |tok| first 40 VAL frames", float(a.max()), "share of entries >= 8:", float((a >= 8).mean()), ">= 16:", float((a >= 16).mean()))
print("fp16 ulp [8,16)", 2.0 ** -7, "[16,32)", 2.0 ** -6, "gap 1.56e-2 =", 1.5625e-2, "= 2^-6")
