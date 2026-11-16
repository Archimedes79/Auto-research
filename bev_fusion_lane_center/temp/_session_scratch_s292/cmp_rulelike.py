import json, os, pickle
import numpy as np
R = os.environ["BFLC_ROOT"] + "/runs/"
def eq(x, y):
    if isinstance(x, dict): return x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)): return len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y))
    return bool(np.array_equal(x, y))
for s in (0, 1):
    d = f"{R}e312_p_s{s}/"
    a = json.load(open(d + "ends_card_snap_thr0.1.json")); b = json.load(open(d + "ends_card_e325rl_snap_thr0.1.json"))
    ca = pickle.load(open(d + "ends_card_snap_thr0.1.counts.pkl", "rb")); cb = pickle.load(open(d + "ends_card_e325rl_snap_thr0.1.counts.pkl", "rb"))
    print(f"seed {s}: numbers identical {a['numbers'] == b['numbers']}, curves {a['curves']} {b['curves']}, wired {a['wired_both_ends']} {b['wired_both_ends']}, per-frame counts identical {eq(ca['head'], cb['head']) and eq(ca['base'], cb['base'])}, wire {b['wire']} {b['wire_run']}")
