"""the new selftest cases against a given tools dir (argv[1]); prints which hold"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
T = tempfile.mkdtemp(prefix="rv3c_")
os.environ["AR_PROJECT"] = T
json.dump(dict(tag_prefix="t", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md",
               code_dirs=["code"], scratch="temp", live_code=["code"], lower_is_better=["vert100"]), open(f"{T}/project.json", "w"))
os.makedirs(f"{T}/experiments/E1_x")
import decide as DC, experiment as X, card as CARD
full = dict(schema=1, split="val", frames=10, commit="selftest", date="2026-10-25 17:00")
base = dict(id="E1", title="t", kind="variant", metric="Q", direction="higher", band=0.01, prediction=dict(value=0.03, vs="c"), control="c",
            rule=[dict(metric="Q", min_effect=0.01)], evaluate=dict(split="val", compare="c"))
def verdict(**kw):
    exp = dict(base, **kw)
    return DC.decide_experiment(exp, T)["verdict"]
def fcard(q, err, lib):
    return dict(full, name="x", metrics=dict(Q=q, err=err), **({"lower_is_better": lib} if lib is not None else {}))
json.dump(dict(a=fcard(0.76, 0.20, ["err"]), b=fcard(0.75, 0.21, ["err"]), ca=fcard(0.73, 0.10, ["err"]), cb=fcard(0.72, 0.11, ["err"]),
               na=fcard(0.76, 0.20, None), nb=fcard(0.75, 0.21, None), nca=fcard(0.73, 0.10, None), ncb=fcard(0.72, 0.11, None),
               xa=fcard(0.76, 0.20, []), xb=fcard(0.75, 0.21, [])), open(f"{T}/dir.json", "w"))
arm = lambda *ks: dict(cards=[f"dir.json#{k}" for k in ks])
errguard = [dict(metric="Q", min_effect=0.01), dict(metric="err", max_loss=0.02)]
r = []
r.append(("CUT declared", verdict(arms=dict(a=arm("a", "b"), c=arm("ca", "cb")), rule=errguard) == "CUT"))
r.append(("KEEP undeclared", verdict(arms=dict(a=arm("na", "nb"), c=arm("nca", "ncb")), rule=errguard) == "KEEP"))
try:
    verdict(arms=dict(a=arm("xa", "xb"), c=arm("ca", "cb")), rule=errguard); r.append(("disagree raises", False))
except ValueError as e:
    r.append(("disagree raises", "disagree" in str(e)))
p = f"{T}/experiments/E1_x/experiment.json"
contra = dict(base, arms=dict(a=arm("a", "b"), c=arm("ca", "cb")), metric="err", direction="higher", rule=[dict(metric="err", min_effect=0.01)])
r.append(("contradicts flagged", any("contradicts the cards" in b for b in X.check(contra, p))))
r.append(("lower ok", not any("contradicts" in b for b in X.check(dict(contra, direction="lower"), p))))
r.append(("validate str", any("lower_is_better" in b for b in CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better="err")))))
r.append(("validate [3]", any("lower_is_better" in b for b in CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better=[3])))))
r.append(("validate ok", CARD.validate(dict(full, name="x", metrics=dict(Q=1.0), lower_is_better=["err"])) == []))
print(r)
