"""throw-away probe: CPU time of the E373 head forward on 4 VAL cache frames (bs 1, 2 threads) and the share of G(i) mass below the dump's floor 0.05"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import jtoken_eval as E  # noqa: E402

torch.set_num_threads(2)
m, args = E.JKT.load_model("e373_cb_s0", "cpu")
ds = E.open_cache("val", 0)
t0 = time.time()
miss, tot = 0.0, 0.0
for b in E.batches(ds, range(4), bs=1):
    out = E.run_head(m, b, "cpu")
    K = out["cand"].shape[-1]
    G = (b["y"] > 0)[:, None] & out["cand"]
    p = out["logp"][..., :K].exp()
    tot += float((p * G).sum())
    miss += float((p * G * (p < 0.05)).sum())
print(f"{(time.time() - t0) / 4:.2f} s per frame; G mass below 0.05: {miss / tot:.4f}")
