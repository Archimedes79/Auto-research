"""One-off: print the train records (losses, Kendall s, gradient diagnostic) and the val records of runs/<name>/log.jsonl."""
import json
import sys

recs = [json.loads(line) for line in open(f"runs/{sys.argv[1]}/log.jsonl")]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 500
for r in recs:
    if "val" in r:
        print("VAL", r["step"], {k: round(v, 4) for k, v in r["val"].items()})
    elif r["step"] % every == 0 or "diag" in r:
        print(r["step"], {k: round(v, 4) for k, v in r.items() if isinstance(v, (int, float)) and k != "step"})
        if "diag" in r:
            d = r["diag"]
            print("   grad norm at BEV feature", dict(zip(d["groups"], [f"{x:.2e}" for x in d["norm"]])),
                  "cos", [[round(c, 2) for c in row] for row in d["cos"]])
