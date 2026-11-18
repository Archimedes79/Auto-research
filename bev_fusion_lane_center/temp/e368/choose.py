"""E368 read-out (scratch): the cell chosen on a run's OWN TRAIN-sample sweep table by the rule of the E368 card (the E366 rule on one table): among the cells with dup_frac <= 0.05 the highest topo F1; exact ties to the cell nearest the
decode of record (link threshold, then keep threshold, then the none gate). usage: python choose.py <run> [<run> ...]  -> prints the cell as 'link keep gate' per run"""
import json
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
rec = (0.3, 0.5, True)
dist = lambda c: (abs(c[0] - rec[0]), abs(c[1] - rec[1]), c[2] != rec[2])  # noqa: E731
for run in sys.argv[1:]:
    t = json.load(open(os.path.join(ROOT, "runs", run, "token_sweep_train_every3.json")))
    rows = [(r["topo_F1"], (r["link_thr"], r["keep_thr"], bool(r["none_gate"])), r["dup_frac"]) for r in t["rows"] if r["dup_frac"] <= 0.05]
    best = max(r[0] for r in rows)
    top = sorted([r for r in rows if abs(r[0] - best) < 1e-12], key=lambda r: dist(r[1]))[0]
    print(run, "cell", top[1][0], top[1][1], "gate_on" if top[1][2] else "gate_off", "| TRAIN topo F1 %.4f dup %.4f | frames %s" % (top[0], top[2], t["frames"]))
