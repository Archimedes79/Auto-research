"""throw-away probes of the lower_is_better change; argv[1] = autoresearch/tools dir"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
T = tempfile.mkdtemp(prefix="rv3_")
os.environ["AR_PROJECT"] = T
json.dump(dict(tag_prefix="lcr", ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", schedule="schedule.csv", test_log="TEST.md",
               code_dirs=["code"], scratch="temp", live_code=["code"]), open(f"{T}/project.json", "w"))
os.makedirs(f"{T}/experiments/E1_x")
import decide as DC, experiment as X
full = dict(schema=1, split="val", frames=10, commit="x", date="2026-10-31 10:00")
LIB = ["false_len_frac", "vert100", "verts_per_frame", "dup_frac"]
def save(d):
    json.dump(d, open(f"{T}/c.json", "w"))
def run(exp):
    p = f"{T}/experiments/E1_x/experiment.json"
    json.dump(exp, open(p, "w"))
    bad = X.check(exp, p)
    try:
        v = DC.decide_experiment(exp, T)
        return bad, v["verdict"], v["why"], [c["text"] for c in v["clauses"]]
    except Exception as e:
        return bad, "EXC", repr(e), []
base = dict(id="E1", title="t", kind="variant", metric="Q", direction="higher", band=0.01, prediction=dict(value=0.03, vs="c"), control="c",
            evaluate=dict(split="val"))
two = dict(a=dict(cards=["c.json#a", "c.json#b"]), c=dict(cards=["c.json#ca", "c.json#cb"]))
# P1: a copied evaluate.py card (inherits lower_is_better) with an added asm_ms; control = timing card without the field. variant 5 ms SLOWER
nc = lambda q, ms, lib=LIB: dict(full, name="v", metrics=dict(Q=q, asm_ms=ms), lower_is_better=lib)
oc = lambda q, ms: dict(full, name="c", metrics=dict(Q=q, asm_ms=ms))
save(dict(a=nc(0.76, 25.0), b=nc(0.75, 25.0), ca=oc(0.73, 20.0), cb=oc(0.72, 20.0)))
print("P1 guard asm_ms, variant slower, mixed new/old:", run(dict(base, arms=two, rule=[dict(metric="Q", min_effect=0.01), dict(metric="asm_ms", max_loss=1.0)])))
save(dict(a=oc(0.76, 25.0), b=oc(0.75, 25.0), ca=oc(0.73, 20.0), cb=oc(0.72, 20.0)))
print("P1b same numbers, all old cards:", run(dict(base, arms=two, rule=[dict(metric="Q", min_effect=0.01), dict(metric="asm_ms", max_loss=1.0)])))
# P2: verts_per_frame guard: old cards vs new cards give different directions
vc = lambda q, v, lib: dict(full, name="v", metrics=dict(Q=q, verts_per_frame=v), **({"lower_is_better": lib} if lib is not None else {}))
for lib in (None, LIB):
    save(dict(a=vc(0.76, 60, lib), b=vc(0.75, 60, lib), ca=vc(0.73, 40, lib), cb=vc(0.72, 40, lib)))
    print("P2 guard verts_per_frame (variant +20 verts), lib", "new" if lib else "old", run(dict(base, arms=two, rule=[dict(metric="Q", min_effect=0.01), dict(metric="verts_per_frame", max_loss=1.0)])))
# P3: subset cards: headline clause on subset n only; subset t holds timing cards of another evaluator that declares [] and has no Q
sub = dict(a=dict(cards={"n": ["c.json#a", "c.json#b"], "t": ["c.json#ta", "c.json#tb"]}), c=dict(cards={"n": ["c.json#ca", "c.json#cb"], "t": ["c.json#tca", "c.json#tcb"]}))
qc = lambda q: dict(full, name="v", metrics=dict(err=q), lower_is_better=["err"])
tc = lambda ms: dict(full, name="v", metrics=dict(ms=ms), lower_is_better=["ms"])
save(dict(a=qc(0.10), b=qc(0.11), ca=qc(0.20), cb=qc(0.21), ta=tc(5.0), tb=tc(5.0), tca=tc(5.0), tcb=tc(5.0)))
print("P3 subset check:", run(dict(base, metric="err", direction="lower", arms=sub, rule=[dict(metric="err", min_effect=0.01, subsets=["n"]), dict(metric="ms", max_loss=1.0, subsets=["t"])])))
# P4: non-headline disagreement: check passes, decide raises
dc = lambda q, e, lib: dict(full, name="v", metrics=dict(Q=q, err=e), lower_is_better=lib)
save(dict(a=dc(0.76, 0.2, []), b=dc(0.75, 0.2, []), ca=dc(0.73, 0.1, ["err"]), cb=dc(0.72, 0.1, ["err"])))
print("P4 guard metric disagreement:", run(dict(base, arms=two, rule=[dict(metric="Q", min_effect=0.01), dict(metric="err", max_loss=0.02)])))
# P5: declaration contradicts project names (vert100 lower by names; card says [] )
save(dict(a=dc(0.76, 0.2, []), b=dc(0.75, 0.2, []), ca=dc(0.73, 0.1, []), cb=dc(0.72, 0.1, [])))
