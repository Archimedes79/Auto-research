import glob, json, os, sys
sys.path.insert(0, "../autoresearch/tools")
import decide as D, experiment as X
n = diff = 0
for f in sorted(glob.glob("experiments/E*/decision*.json")):
    old = json.load(open(f))
    ef = f.replace("decision", "experiment", 1) if os.path.exists(f.replace("decision", "experiment", 1)) else os.path.join(os.path.dirname(f), "experiment.json")
    try:
        exp = json.load(open(ef))
        new = D.decide_experiment(exp, os.getcwd())
    except Exception as e:
        print("SKIP", f, type(e).__name__, str(e)[:80]); continue
    n += 1
    if new["verdict"] != old.get("verdict"):
        diff += 1
        print("CHANGED", f, old.get("verdict"), "->", new["verdict"], new["why"][:120])
print(n, "decisions re-run,", diff, "changed")
