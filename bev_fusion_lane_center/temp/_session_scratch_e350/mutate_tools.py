"""mutation check of the selftests of evaluation/jwire_verdict.py and evaluation/jwire_check.py (source text mutated in memory, exec'd; the files are not touched)
usage: bash tools/env/_py14.sh temp/_session_scratch_e350/mutate_tools.py"""
import os
import sys
import traceback
import types

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/evaluation")
V, C = R + "/evaluation/jwire_verdict.py", R + "/evaluation/jwire_check.py"
SRC = {V: open(V, encoding="utf-8").read(), C: open(C, encoding="utf-8").read()}
MUT = [
    ("V1 kill (b) ignored", V, 'bad = [s for s in SEEDS if data[s]["wa"].get("kill_on_val")]', "bad = []"),
    ("V2 a TRAIN-only fire is a kill", V, 'bad = [s for s in SEEDS if data[s]["wa"].get("kill_on_val")]', 'bad = [s for s in SEEDS if data[s]["wa"].get("kill_on_val") or data[s]["wa"].get("kill_on_train")]'),
    ("V3 KEEP- removed", V, "if any(i[2] < 0 for i in iv):", "if False:"),
    ("V4 KEEP+ needs one seed only", V, "if all(i[1] > 0 for i in iv):", "if any(i[1] > 0 for i in iv):"),
    ("V5 topo bound 0.05", V, "BOUND = dict(topo_F1=0.005,", "BOUND = dict(topo_F1=0.05,"),
    ("V6 dup_frac is not a rise", V, 'RISE = ("dup_frac", "vert100_dp")', 'RISE = ("vert100_dp",)'),
    ("V7 shift limit 0.5", V, "SHIFT_LIMIT = 0.02, 0.7", "SHIFT_LIMIT = 0.02, 0.5"),
    ("V8 INCOMPLETE ignored", V, "    if missing:\n", "    if False:\n"),
    ("V9 provenance not checked", V, 'if wa.get("pair_run", data[s]["run"]) != data[s]["run"] or wa.get("seed", s) != s:', "if False:"),
    ("V10 non-finite not checked", V, "    bad = [s for s in SEEDS if not finite(data[s][\"wa\"]) or not finite(data[s][\"w\"][0])]", "    bad = []"),
    ("V11 a missed guard is no TIE", V, "    if any(miss.values()):", "    if False:"),
    ("V12 shift flag note lost", V, 'if wa.get("kill_on_train") and not wa.get("kill_on_val"):', "if False:"),
    ("V13 arm x never printed", V, "        if a.tag_x:\n            x = sd", "        if False:\n            x = sd"),
    ("C1 any one-row difference is a pair", C, 'if all(_close(np.asarray(big[k])[keep], small[k], atol) for k in ("pairs", "kp_p", "gts")):', "if True:"),
    ("C2 two frames tolerated", C, "len(diff) == 1 and list(diff.values())[0] == [\"one pair\"]", "len(diff) <= 2 and all(v == [\"one pair\"] for v in diff.values())"),
    ("C3 ends / attrs not compared", C, 'FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")', 'FIELDS = ("pairs", "kp_p", "gts")'),
    ("C4 numbers of the cards ignored", C, 'IGNORED = ("tag", "wire", "wire_run")', 'IGNORED = ("tag", "wire", "wire_run", "numbers")'),
    ("C5 per-frame counts of the head ignored", C, "return sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))", 'return sorted(k for k in set(a) | set(b) if k != "head" and not same(a.get(k), b.get(k)))'),
    ("C6 a head card passes as the twin", C, 'if twin.get("wire") != "twin" or twin.get("tag") != tag:', "if False:"),
    ("C7 seed of the cards not checked", C, 'if c["run"] != run or c["seed"] != seed:', "if False:"),
    ("C8 a tolerance in the exact comparison", C, "    return x == y or (x != x and y != y)", "    return (isinstance(x, float) and isinstance(y, float) and abs(x - y) < 1e-6) or x == y or (x != x and y != y)"),
    ("C9 tolerance 0.5", C, "ATOL = 2e-3", "ATOL = 0.5"),
    ("C10 no frame at all is accepted", C, "    ok = bool(live) and (not diff", "    ok = (not diff"),
]


def load(name, path, text):
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(text, path, "exec"), mod.__dict__)
    return mod


res = []
for name, path, old, new in MUT:
    t = SRC[path]
    if t.count(old) != 1:
        res.append((name, "BAD MUTANT", f"text occurs {t.count(old)} times"))
        continue
    try:
        load("m", path, t.replace(old, new))._selftest()
        res.append((name, "SURVIVED", ""))
    except BaseException as e:
        res.append((name, "killed", f"{type(e).__name__}: {str(e)[:90]}"))
print("\n=== mutation result ===")
for n, s, d in res:
    print(f"{s:10s} {n:45s} {d}")
print(f"killed {sum(s == 'killed' for _, s, _ in res)} / {len(res)}")
