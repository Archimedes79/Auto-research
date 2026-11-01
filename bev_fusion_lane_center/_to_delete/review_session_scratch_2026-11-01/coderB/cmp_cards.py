"""every metric of the golden card (old code) against the golden_check card (new code), per registry name"""
import json, os, glob
R = os.environ["BFLC_ROOT"]
for new in sorted(glob.glob(f"{R}/runs/eval/*/val_golden_check.json")):
    old = new.replace("_golden_check.json", "_golden.json")
    if not os.path.exists(old):
        print("no golden card for", new); continue
    a, b = json.load(open(old)), json.load(open(new))
    ma, mb = a["metrics"], b["metrics"]
    diff = [(k, ma[k], mb.get(k)) for k in sorted(ma) if isinstance(ma[k], (int, float)) and isinstance(mb.get(k), (int, float)) and abs(ma[k] - mb[k]) > 1e-9]
    name = os.path.basename(os.path.dirname(new))
    print(f"{name}: old commit {a.get('commit')} frames {a.get('frames')} | new commit {b.get('commit')} frames {b.get('frames')} | {len(diff)} of {len(ma)} metrics differ")
    for k, x, y in diff:
        print(f"    {k:22s} {x:.4f} -> {y:.4f}  ({y - x:+.4f})")
