"""Scratch (read-only): mean of the logged scale_l1 over the steps 2100 - 3000 (ten rows of one batch each) for E312 and E320, and the log row spacing.  usage: python3 temp/e317/_scale_l1.py"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
for run in ("e312_p_s0", "e312_p_s1", "e320_a_s0", "e320_a_s1"):
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "runs", run, "log.jsonl")) if l.strip()]
    sel = [r["scale_l1"] for r in rows if 2100 <= r["step"] <= 3000 and "scale_l1" in r]
    steps = [r["step"] for r in rows]
    print(f"{run}: rows {len(rows)} (steps {steps[0]} .. {steps[-1]}, every {steps[1] - steps[0]}), scale_l1 mean over steps 2100-3000 {sum(sel) / len(sel):.4f} ({len(sel)} rows), geom_sig in log: {any('geom_sig' in r for r in rows)}")
