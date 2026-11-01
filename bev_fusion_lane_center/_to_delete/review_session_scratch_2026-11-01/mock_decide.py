import json, os, shutil, sys, glob, random
sys.path.insert(0, "../autoresearch/tools")
SP = sys.argv[1]
root = os.path.join(SP, "mockproj")
shutil.rmtree(root, ignore_errors=True)
os.makedirs(root)
shutil.copy("project.json", root)
base = json.load(open("runs/eval/recall/val.json"))
def card(path, name, bump, cmp=None):
    m = dict(base["metrics"]); m["R@1_far_inner"] = 0.80; m["R@1_far_ring"] = 0.70
    for k, v in bump.items(): m[k] = m.get(k, 0) + v
    c = dict(schema=1, commit="x", date="2026-11-01", name=name, split="val", subset="", fresh=False, frames=873, metrics=m, better=base.get("better", {}), protocol=dict(gt_sha="abc"))
    if "better" not in base or not base.get("better"):
        c.pop("better")
    if cmp:
        c["compare"] = dict(name=cmp, metrics=dict(base["metrics"]), paired={k: dict(diff=v, lo=v-0.004, hi=v+0.004) for k, v in bump.items()})
    full = os.path.join(root, path); os.makedirs(os.path.dirname(full), exist_ok=True)
    json.dump(c, open(full, "w"))
for e in ["E265", "E267", "E264", "E266", "E261", "E262"]:
    src = glob.glob(f"experiments/{e}_*/experiment.json")[0]
    d = os.path.join(root, "experiments", os.path.basename(os.path.dirname(src))); os.makedirs(d)
    shutil.copy(src, d)
    exp = json.load(open(src))
    for a, arm in exp["arms"].items():
        refs = [f"runs/eval/{n}/val.json" for n in arm.get("names", [])] + list(arm.get("cards", []) if isinstance(arm.get("cards"), list) else [])
        for i, r in enumerate(refs):
            nm = r.split("/")[2]
            good = a == exp.get("variant")
            bump = {"lanes_found": 0.02, "R@1_far": 0.02} if good else ({"lanes_found": 0.01, "R@1_far": 0.005} if a in ("s", "r", "t", "m", "b") else {})
            ctrl_name = None
            if good:
                cref = exp["arms"][exp["control"]]
                cl = [f"runs/eval/{n}/val.json" for n in cref.get("names", [])] + list(cref.get("cards", []))
                ctrl_name = cl[min(i, len(cl)-1)].split("/")[2]
            card(r, nm, bump, ctrl_name)
os.environ["AR_PROJECT"] = root
import experiment as X, decide as D
for f in sorted(glob.glob(os.path.join(root, "experiments", "*", "experiment.json"))):
    exp = json.load(open(f))
    bad = X.check(exp, f)
    if bad:
        print(exp["id"], "NOT READY", bad); continue
    r = D.decide_experiment(exp, root)
    print(exp["id"], r["verdict"], r["why"][:150])
