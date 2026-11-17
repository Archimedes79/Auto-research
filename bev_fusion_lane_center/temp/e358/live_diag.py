"""E358 diagnostic (scratch, GPU, no verdict): the live pass of the attached-end records (batch size 4, as `E358.live_s0`) against the stored-dump records, with the MAGNITUDE of every difference: per frame and field the max |difference| and the number of rows above atol 2e-3.   usage via run.py: bash tools/env/_pyr.sh live_diag.py --seed 0 [--bs 4]"""
import argparse
import os
import sys
import numpy as np
ROOT = os.environ["BFLC_ROOT"]
for sub in ("evaluation", "training", "training/grid", "training/data", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import jwire_data as JD  # noqa: E402
import jwire_check as JC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--max", type=int, default=24)
a = ap.parse_args()
run = f"e324_s_s{a.seed}"
ref = {r["row"]: r for r in JD.load_records(JD.cache_path("val", a.seed, pair_run=run, ends="att"), a.seed, run, "att")}
live = JD.live_records(a.seed, "val_clean", 0.1, a.max, 2, bs=a.bs, device="cuda", pair_run=run, ends="att")
print(f"seed {a.seed} batch size {a.bs}: {len(live)} live frames")
for r in live:
    q = ref[r["row"]]
    out = []
    for k in JC.FIELDS:
        x, y = np.asarray(r[k], float), np.asarray(q[k], float)
        if x.shape != y.shape:
            out.append(f"{k}: SHAPE {x.shape} vs {y.shape}")
            continue
        d = np.abs(x - y)
        d = np.where(np.isnan(x) & np.isnan(y), 0.0, d)
        if d.size and d.max() > JC.ATOL:
            rows = np.unique(np.nonzero(d > JC.ATOL)[0]) if d.ndim > 1 else np.nonzero(d > JC.ATOL)[0]
            out.append(f"{k}: max {d.max():.4f}, {len(rows)} of {len(x)} rows above {JC.ATOL}")
    print(f"row {r['row']}: " + ("; ".join(out) if out else "equal"))
