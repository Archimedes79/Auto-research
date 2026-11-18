import os, sys, pickle
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "evaluation")); sys.path.insert(0, os.path.join(ROOT, "training/data"))
import torch
import lane_ptok_data as LP, lane_tokcache as LC
import jtoken_eval as JE, token_readout as TR
for run in ("e367_b25_s0", "e365_s1_s0"):
    d = pickle.load(open(os.path.join(ROOT, "runs", run, "token_val.pkl"), "rb"))
    c = LC.TokCache(LP.cache_dir("val", 0, "e292_a")); tok2i = {t: i for i, t in enumerate(c.tokens)}
    s, y = [], []
    for f in d["frames"]:
        i = tok2i[f["token"]]; v = np.array(c.f["valid"][i])
        s.append(1.0 - np.asarray(f["pnone"], float)[v]); y.append(np.array(c.f["gt_id"][i])[v] >= 0)
    s, y = np.concatenate(s), np.concatenate(y)
    ties = len(s) - len(np.unique(s))
    print(run, "n", len(y), "tied scores", ties, "sklearn AP %.6f" % TR.existence_ap(s, y), "pessimistic AP %.6f" % JE.average_precision(s, y))
