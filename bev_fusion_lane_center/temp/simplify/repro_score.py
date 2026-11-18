"""D117 simplification (scratch): the reproduction check of the lead's criterion: `score` on the stored champion head e373_cb_s0 at the D115 cell must give the
card's numbers bit for bit before and after each unit. Runs the CURRENT jtoken_eval.score on a copy of the run folder (model.pt + token_val.pkl) under
temp/simplify/runs_<tag>/ (nothing is written under runs/), then compares every metric with the stored card runs/e373_cb_s0/token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json.
usage: bash tools/env/_py14.sh repro_score.py <tag>"""

import json
import os
import shutil
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import jtoken_eval as E  # noqa: E402

tag = sys.argv[1]
run, seed = "e373_cb_s0", 0
src = os.path.join(ROOT, "runs", run)
dst_root = os.path.join(ROOT, "temp", "simplify", f"runs_{tag}")
dst = os.path.join(dst_root, run)
assert not os.path.exists(dst), f"{dst} exists: use a new tag"
os.makedirs(dst)
for f in ("model.pt", "token_val.pkl", "config.json"):
    shutil.copy2(os.path.join(src, f), dst)
a = E.parse_args(["score", "--run", run, "--seed", str(seed), "--link-thr", "0.3", "--link-thr-long", "0.2", "--keep-thr", "0.6", "--no-none-gate"])
out = E.score(a, runs=dst_root)
stored = json.load(open(os.path.join(src, "token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json")))["metrics"]
new = json.load(open(os.path.join(dst, "token_card_rec_lt0.3_ltl0.2_kt0.6_nng.json")))["metrics"]
diff = {k: (stored[k], new.get(k)) for k in stored if stored[k] != new.get(k)}
print(f"[{tag}] metrics {len(stored)}; differing: {len(diff)}")
for k, v in list(diff.items())[:20]:
    print("  ", k, v)
print(f"[{tag}] topo_F1 stored {stored['topo_F1']!r} new {new.get('topo_F1')!r}; junction_F1 stored {stored['junction_F1']!r} new {new.get('junction_F1')!r}")
print(f"[{tag}] BIT-FOR-BIT" if not diff else f"[{tag}] DIFFERS")
