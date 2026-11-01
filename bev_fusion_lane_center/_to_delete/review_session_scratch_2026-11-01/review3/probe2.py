"""throw-away probe: check counts cards that do not hold the headline metric; argv[1] = autoresearch/tools dir"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
T = tempfile.mkdtemp(prefix="rv3b_")
os.environ["AR_PROJECT"] = T
json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md",
               code_dirs=["code"], scratch="temp", live_code=["code"]), open(f"{T}/project.json", "w"))
os.makedirs(f"{T}/experiments/E1_x")
import decide as DC, experiment as X
full = dict(schema=1, split="val", frames=10, commit="x", date="2026-10-31 10:00")
qc = lambda e: dict(full, name="v", metrics=dict(err=e), lower_is_better=["err"])
tc = lambda ms: dict(full, name="v", metrics=dict(ms=ms), lower_is_better=["ms"])
json.dump(dict(a=qc(0.10), b=qc(0.11), ca=qc(0.20), cb=qc(0.21), ta=tc(5.0), tb=tc(5.0), tca=tc(5.0), tcb=tc(5.0)), open(f"{T}/c.json", "w"))
arms = dict(a=dict(cards={"": ["c.json#a", "c.json#b"], "t": ["c.json#ta", "c.json#tb"]}), c=dict(cards={"": ["c.json#ca", "c.json#cb"], "t": ["c.json#tca", "c.json#tcb"]}))
exp = dict(id="E1", title="t", kind="variant", metric="err", direction="lower", band=0.01, prediction=dict(value=0.1, vs="c"), control="c",
           evaluate=dict(split="val"), arms=arms, rule=[dict(metric="err", min_effect=0.01, subsets=[""]), dict(metric="ms", max_loss=1.0, subsets=["t"])])
p = f"{T}/experiments/E1_x/experiment.json"
json.dump(exp, open(p, "w"))
print("check (all cards consistent, timing subset has no err):", X.check(exp, p))
d = DC.decide_experiment(exp, T)
print("decide:", d["verdict"], d["why"])
# only the timing cards declare (whole-split cards are old, no field): headline lower -> check says 'contradicts'
for k in ("a", "b", "ca", "cb"):
    pass
c = json.load(open(f"{T}/c.json"))
for k in ("a", "b", "ca", "cb"):
    c[k].pop("lower_is_better")
json.dump(c, open(f"{T}/c.json", "w"))
print("check (old err cards, declaring timing cards):", X.check(exp, p))
