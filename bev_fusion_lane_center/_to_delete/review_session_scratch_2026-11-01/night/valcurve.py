"""validation loss rows of bev runs' log.jsonl: is the run still improving at its last step?"""
import json
import sys

for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"/mnt/c/Auto-research/bev_fusion/runs/{run}/log.jsonl") if '"val"' in l]
    print(run)
    for r in rows:
        v = r["val"]
        print(f"  step {r['step']:5d}  total {v.get('total', float('nan')):.3f}  areas {v.get('group.areas', float('nan')):.3f}  lines {v.get('group.lines', float('nan')):.3f}"
              f"  objects {v.get('group.objects', float('nan')):.3f}  motion {v.get('group.motion', float('nan')):.3f}")
