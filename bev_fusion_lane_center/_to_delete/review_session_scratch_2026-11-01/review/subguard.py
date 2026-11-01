import json, os, sys, tempfile
sys.path.insert(0, "/mnt/c/Auto-research/autoresearch/tools")
T = tempfile.mkdtemp(prefix="rv_")
json.dump(dict(tag_prefix="t", lower_is_better=[]), open(T + "/project.json", "w"))
os.environ["AR_PROJECT"] = T
json.dump(dict(n=dict(a=dict(R=0.80), c=dict(R=0.70)), b=dict(a=dict(R=0.80), c=dict(R=0.70)), x=dict(a=dict(R=0.69), c=dict(R=0.70))), open(T + "/sub.json", "w"))
import decide as DC
arms = dict(a=dict(cards={s: f"sub.json#{s}/a" for s in "nbx"}), c=dict(cards={s: f"sub.json#{s}/c" for s in "nbx"}))
for det in (False, True):
    exp = dict(id="E9", metric="R", direction="higher", band=0.01, control="c", arms=arms, deterministic=det,
               rule=[dict(metric="R", max_loss=0.05, subsets=["n", "b", "x"], at_least=2)], prediction=dict(value=0.05, vs="c"), evaluate=dict(split="val"))
    d = DC.decide_experiment(exp, T)
    print("deterministic", det, d["verdict"], d["why"])
# lower-is-better via project key: a guard on a non-headline error metric in a project with an empty list
json.dump(dict(a=dict(Q=0.80, ate=0.60), b=dict(Q=0.81, ate=0.61), ca=dict(Q=0.70, ate=0.40), cb=dict(Q=0.71, ate=0.41)), open(T + "/c.json", "w"))
arms2 = dict(a=dict(cards=["c.json#a", "c.json#b"]), c=dict(cards=["c.json#ca", "c.json#cb"]))
exp = dict(id="E8", metric="Q", direction="higher", band=0.01, control="c", arms=arms2, rule=[dict(metric="Q", min_effect=0.01), dict(metric="ate", max_loss=0.02)],
           prediction=dict(value=0.05, vs="c"), evaluate=dict(split="val"))
print("ate +0.20 worse, empty lower_is_better:", DC.decide_experiment(exp, T)["verdict"])
