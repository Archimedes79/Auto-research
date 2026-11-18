"""E379 scratch probe (one use): how many links of the ensemble decode can the dumps' floor (candidates p >= 0.05 only) move? A pair that one head wrote and the other did not gets the mean p_A / 2 although the true mean is
(p_A + q) / 2 with q < 0.05: the ensemble's p is low by less than 0.025 there. Counted at the D115 cell (short 0.3 / long 0.2, keep 0.6, gate off): the one-sided pairs with both ends kept whose truncated mean lies in
[T - 0.025, T) (the links a full-probability ensemble could add, an upper bound), against the delivered links. usage: e379e_floor_probe.py <seed>"""
import argparse
import os
import sys

import numpy as np

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "temp", "e379", "e379e_session_scratch_dev", "evaluation"))     # the dev copy (mean_frame)
import jtoken_eval as JE  # noqa: E402

S = int(sys.argv[1])
LT, LTL, KT = 0.3, 0.2, 0.6
ns = lambda run: argparse.Namespace(run=run, model="model.pt", seed=S, split="val", every=1, drop=0.0, drop_seed=0, live=False)  # noqa: E731
ra, rb = (JE.read_dump(ns(x), JE.RUNS, 0) for x in (f"e373_cb_s{S}", f"e367_b25_s{S}"))
JE.same_tokens(ra, rb)
risk = deliv = one_sided = both = 0
for fa, fb, v in zip(ra["fr"], rb["fr"], ra["valid"]):
    e = JE.mean_frame(fa, fb)
    K = len(e["pnone"])
    key = lambda c: set((c["src"].astype(int) * K + c["dst"].astype(int)).tolist())  # noqa: E731
    ka, kb = key(fa["cand"]), key(fb["cand"])
    c = e["cand"]
    k = c["src"].astype(int) * K + c["dst"].astype(int)
    one = np.array([(x in ka) != (x in kb) for x in k], bool)
    keep = v & (e["pnone"] < np.float32(KT))
    ends = keep[c["src"].astype(int)] & keep[c["dst"].astype(int)]
    thr = np.where(c["long"], np.float32(LTL), np.float32(LT))
    risk += int((one & ends & (c["p"] < thr) & (c["p"] >= thr - JE.KEEP_P / 2)).sum())
    deliv += int((ends & (c["p"] >= thr)).sum())
    one_sided += int(one.sum())
    both += int((~one).sum())
n = len(ra["fr"])
print(f"seed {S}, VAL {n} frames, D115 cell: candidates of the ensemble per frame {(one_sided + both) / n:.1f} (one-sided {one_sided / n:.1f}); delivered links per frame {deliv / n:.2f}; "
      f"one-sided pairs with both ends kept within 0.025 under their class threshold (could be links with the full probabilities) per frame {risk / n:.3f} = {risk / max(deliv, 1):.4f} of the delivered links")
