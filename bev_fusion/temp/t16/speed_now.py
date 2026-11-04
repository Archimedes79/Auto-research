"""scratch: seconds per step of a training run over its last log entries (wall clock between log rows). usage: speed_now.py t18_fix [n]"""
import json
import sys

rows = [json.loads(l) for l in open(f"bev_fusion/runs/{sys.argv[1]}/log.jsonl", encoding="utf-8") if l.startswith("{")]
rows = [r for r in rows if "t" in r and "step" in r]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
for a, b in zip(rows[-n - 1:-1], rows[-n:]):
    print(f"step {a['step']}->{b['step']}: {(b['t'] - a['t']) / (b['step'] - a['step']):.2f} s/step   (logged s_per_step {b.get('s_per_step', 0):.2f})")
import time
print("last row", round(time.time() - rows[-1]["t"]), "s ago")
