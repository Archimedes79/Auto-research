import glob, json, os
from collections import Counter
ROOT = "/mnt/c/Auto-research/bev_fusion"
cf = {}
for p in sorted(glob.glob(f"{ROOT}/training/configs/*.json") + glob.glob(f"{ROOT}/runs/*/config.json")):
    try: c = json.load(open(p, encoding="utf-8"))
    except Exception: continue
    n = os.path.basename(os.path.dirname(p)) if "/runs/" in p else os.path.splitext(os.path.basename(p))[0]
    cf.setdefault(n, c)
for k in ("balancing", "grad_diag_every", "consistency", "radar_aux", "probe", "seq_len", "seq_bptt"):
    cnt = Counter(json.dumps(c.get(k)) for c in cf.values() if isinstance(c, dict))
    print(k, dict(cnt))
print([n for n, c in cf.items() if isinstance(c, dict) and c.get("balancing") == "uwso"], [n for n, c in cf.items() if isinstance(c, dict) and (c.get("consistency") or 0) > 0])
print([n for n, c in cf.items() if isinstance(c, dict) and (c.get("grad_diag_every") or 0) > 0][:12])
