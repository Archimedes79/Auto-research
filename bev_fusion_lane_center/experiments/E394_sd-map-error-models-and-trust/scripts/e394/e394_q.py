import json, sys, os
ROOT = os.environ.get("BFLC_ROOT") or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))   # the problem folder (this copy lies in experiments/E394_*/scripts/e394/; it ran from temp/)
import numpy as np
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import maptrust_gate as G
rows = json.load(open(os.path.join(ROOT, "temp", "e394_smoke", "explore_rows_s0.json")))
X = np.array([r["x"][:4] for r in rows]); ws = np.array([r["ws"] for r in rows])
n, c1, cz = (np.array([r[k] for r in rows]) for k in ("n", "c1", "cz"))
f = G.fit_logistic(X, ws); p = G.gate_prob(f, X)
for q in (0.90, 0.925, 0.95, 0.96, 0.975, 0.99):
    thr = np.quantile(p[ws == 0], q); ok = p <= thr
    acc = lambda sel: np.where(ok[sel], c1[sel], cz[sel]).sum() / n[sel].sum()
    print(f"q {q}: trust rest {ok[ws==0].mean():.3f} rejects ws {1-ok[ws==1].mean():.3f} | ws g - z {acc(ws==1) - cz[ws==1].sum()/n[ws==1].sum():+.3f} | rest g {acc(ws==0):.3f} (a {c1[ws==0].sum()/n[ws==0].sum():.3f}, z {cz[ws==0].sum()/n[ws==0].sum():.3f})")
specs = {r["k"]: G.ME.train_spec(np.random.default_rng([394, r["k"], 1])) for r in rows}
simple_keys = {"pose", "count_rate", "drop", "hide", "node_drop"}
cat = np.array(["simple" if set(specs[r["k"]]) <= simple_keys else ("ws" if specs[r["k"]].get("wrong_scene") else "other") for r in rows])
for q in (0.90, 0.925, 0.95, 0.96, 0.975):
    thr = np.quantile(p[ws == 0], q); ok = p <= thr
    print(f"Q {q}: trust on simple/exact draws {ok[cat=='simple'].mean():.3f} (n {int((cat=='simple').sum())}), on other non-ws {ok[cat=='other'].mean():.3f} (n {int((cat=='other').sum())})")
