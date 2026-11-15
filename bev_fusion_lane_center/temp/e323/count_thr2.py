import os, pickle
import numpy as np
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
def P(run):
    fr = pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))
    return [1.0 / (1.0 + np.exp(-f["logit"])) * f["allowed"] for f in fr]
ts = np.round(np.arange(0.03, 0.31, 0.001), 3)
for S, tgt in ((0, 12018), (1, 12208)):
    ps = P(f"e312_p_s{S}")
    cnt = np.array([sum(int((p >= t).sum()) for p in ps) for t in ts])
    j = int(np.argmin(np.abs(cnt - tgt)))
    print(f"seed {S}: E312 delivers {cnt[j]} curves at threshold {ts[j]:.3f} (target = E320's {tgt} at p >= 0.1: {100 * (cnt[j] - tgt) / tgt:+.2f} %)")
