"""scratch: which runs have logged val rows, and how the val numbers move with the step (answer to: does it get worse after 12000 steps?)"""
import glob, json, os
rows = []
for p in sorted(glob.glob("/mnt/c/Auto-research/bev_fusion/runs/*/log.jsonl")):
    run = os.path.basename(os.path.dirname(p))
    last = 0
    vals = []
    try:
        with open(p) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                s = r.get("step", 0)
                last = max(last, s)
                if "val" in r:
                    vals.append((s, r["val"]))
    except Exception as e:
        continue
    rows.append((run, last, vals))
print("runs with max step >= 6000 or val rows:")
for run, last, vals in rows:
    if last >= 5000 or len(vals) >= 3:
        print(f"{run:34s} last step {last:6d}  val rows at steps {[s for s, _ in vals][:12]}")
# keys of one val row
for run, last, vals in rows:
    if vals:
        print("example val keys:", run, list(vals[0][1].keys())[:25])
        break
