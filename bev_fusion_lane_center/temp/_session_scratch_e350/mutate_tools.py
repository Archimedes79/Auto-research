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
    ("V1 kill (b) ignored", V, "    bad = [s for s in SEEDS if kv[s]]", "    bad = []"),
    ("V2 a TRAIN-only fire is a kill", V, "        kv[s], kt = kills(wa)", "        kv[s], kt = kills(wa)[0] or kills(wa)[1], False"),
    ("V3 KEEP- removed", V, 'if any(i["upper"] < 0 for i in iv):', "if False:"),
    ("V4 KEEP+ needs one seed only", V, 'if all(i["lower"] > 0 for i in iv):', 'if any(i["lower"] > 0 for i in iv):'),
    ("V5 topo bound 0.05", V, "BOUND = dict(topo_F1=0.005,", "BOUND = dict(topo_F1=0.05,"),
    ("V5b junction bound 0.1", V, "junction_F1=0.01, lanes_found=0.01)", "junction_F1=0.1, lanes_found=0.01)"),
    ("V6 dup_frac is not a rise", V, 'RISE = ("dup_frac", "vert100_dp")', 'RISE = ("vert100_dp",)'),
    ("V6b guards: a rise is a fall", V, "((d[k] if k in RISE else -d[k]) <= BOUND[k])", "((-d[k]) <= BOUND[k])"),
    ("V6c a guard dropped from K (junction F1)", V, 'K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")', 'K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "vert100_dp")'),
    ("V7 shift limit 0.5", V, "KILL_MARGIN, SHIFT_LIMIT, POINT_TOL = 0.02, 0.7, 1e-4", "KILL_MARGIN, SHIFT_LIMIT, POINT_TOL = 0.02, 0.5, 1e-4"),
    ("V8 INCOMPLETE ignored", V, "    if missing:\n", "    if False:\n"),
    ("V9 pair_run not required", V, 'if "pair_run" not in wa and run != f"e312_p_s{s}":', "if False:"),
    ("V9b pair_run of another run accepted", V, 'if wa.get("pair_run", run) != run or wa.get("seed") != s:', 'if wa.get("seed") != s:'),
    ("V9c seed of the head not checked", V, 'if wa.get("pair_run", run) != run or wa.get("seed") != s:', 'if wa.get("pair_run", run) != run:'),
    ("V10 non-finite not checked", V, '    bad = [s for s in SEEDS if not finite(data[s]["wa"]) or not finite(data[s]["w"][0])]', "    bad = []"),
    ("V11 a missed guard is no TIE", V, "    if any(miss.values()):", "    if False:"),
    ("V12 shift flag note lost", V, "        if kt and not kv[s]:", "        if False:"),
    ("V13 arm x never printed", V, "        if a.tag_x:\n            x = sd", "        if False:\n            x = sd"),
    ("V14 the stored kill flag is trusted", V, "        kv[s], kt = kills(wa)", '        kv[s], kt = bool(wa["kill_on_val"]), bool(wa["kill_on_train"])'),
    ("V15 no interval: a bare KEEP", V, 'return "INCOMPLETE", "the sign class is open', 'return "KEEP", "the sign class is open'),
    ("V16 the first block of the log counts", V, "    h = heads[-1]", "    h = heads[0]"),
    ("V17 the log header is not checked", V, 'if bt["a"] != want_a or bt["b"] != want_b:', "if False:"),
    ("V17b the header: only A is checked", V, 'if bt["a"] != want_a or bt["b"] != want_b:', 'if bt["a"] != want_a:'),
    ("V17c the header: only B is checked", V, 'if bt["a"] != want_a or bt["b"] != want_b:', 'if bt["b"] != want_b:'),
    ("V18 the point estimate is not checked", V, 'if w0 is not None and abs(bt["point"]', 'if False and abs(bt["point"]'),
    ("V18b the point tolerance 1e-2", V, "KILL_MARGIN, SHIFT_LIMIT, POINT_TOL = 0.02, 0.7, 1e-4", "KILL_MARGIN, SHIFT_LIMIT, POINT_TOL = 0.02, 0.7, 1e-2"),
    ("V19 card run not checked", V, '("run", c.get("run") == run), ', '("run", True), '),
    ("V19b card seed not checked", V, '("seed", c.get("seed") == s), ', '("seed", True), '),
    ("V19c card snap not checked", V, '("snap", c.get("snap") is True), ', '("snap", True), '),
    ("V19d card attach not checked", V, '("attach", not c.get("attach")), ', '("attach", True), '),
    ("V19e card threshold not checked", V, '("thr", c.get("thr") == 0.1)', '("thr", True)'),
    ("V20 control scored with a head accepted", V, "    if c.get(\"wire\") is not None or c.get(\"tag\") is not None:", "    if False:"),
    ("V20b control tag not checked", V, "    if c.get(\"wire\") is not None or c.get(\"tag\") is not None:", "    if c.get(\"wire\") is not None:"),
    ("V21 head wire not checked", V, 'c.get("wire") != "head" or c.get("wire_run") != wire_run or c.get("tag") != tag', 'c.get("wire_run") != wire_run or c.get("tag") != tag'),
    ("V21b head wire_run not checked", V, 'c.get("wire") != "head" or c.get("wire_run") != wire_run or c.get("tag") != tag', 'c.get("wire") != "head" or c.get("tag") != tag'),
    ("V21c head tag not checked", V, 'c.get("wire") != "head" or c.get("wire_run") != wire_run or c.get("tag") != tag', 'c.get("wire") != "head" or c.get("wire_run") != wire_run'),
    ("V22 the model_h suffix of heads 1 and 2 not required", V, 'f"{a.head_prefix}{s}" + (f":model_h{h}.pt" if h else ""))', 'f"{a.head_prefix}{s}" + "")'),
    ("V23 arm x wire_run: the head prefix", V, 'f"{a.x_head_prefix}{s}"))', 'f"{a.head_prefix}{s}"))'),
    ("V24 first head of wire_agreement.json", V, 'if [h["head"] for h in wa["heads"]][:1] != [0] or wa["heads"][0]["ablation"]:', "if False:"),
    ("C1 any one-row difference is a pair", C, 'if all(_close(np.asarray(big[k])[keep], small[k], atol) for k in ("pairs", "kp_p", "gts")):', "if True:"),
    ("C2 two frames tolerated", C, "len(diff) == 1 and list(diff.values())[0] == [\"one pair\"]", "len(diff) <= 2 and all(v == [\"one pair\"] for v in diff.values())"),
    ("C3 ends / attrs not compared", C, 'FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")', 'FIELDS = ("pairs", "kp_p", "gts")'),
    ("C3b kp_p not compared", C, 'FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")', 'FIELDS = ("ends", "attrs", "pairs", "gts")'),
    ("C3c gts not compared", C, 'FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")', 'FIELDS = ("ends", "attrs", "pairs", "kp_p")'),
    ("C4 numbers of the cards ignored", C, 'IGNORED = ("tag", "wire", "wire_run")', 'IGNORED = ("tag", "wire", "wire_run", "numbers")'),
    ("C5 per-frame counts of the head ignored", C, "return sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))", 'return sorted(k for k in set(a) | set(b) if k != "head" and not same(a.get(k), b.get(k)))'),
    ("C5b per-frame counts of the base ignored", C, "return sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))", 'return sorted(k for k in set(a) | set(b) if k != "base" and not same(a.get(k), b.get(k)))'),
    ("C5c scenes of the counts ignored", C, "return sorted(k for k in set(a) | set(b) if not same(a.get(k), b.get(k)))", 'return sorted(k for k in set(a) | set(b) if k != "scenes" and not same(a.get(k), b.get(k)))'),
    ("C6 a head card passes as the twin", C, 'if twin.get("wire") != "twin" or twin.get("tag") != tag:', "if False:"),
    ("C6b the twin's tag not checked", C, 'if twin.get("wire") != "twin" or twin.get("tag") != tag:', 'if twin.get("wire") != "twin":'),
    ("C6c a stored card scored with a head", C, 'if stored.get("wire") is not None:', "if False:"),
    ("C7 run of the cards not checked", C, '        if c["run"] != run:', "        if False:"),
    ("C7b seed of the cards not checked", C, '        if c["seed"] != seed:', "        if False:"),
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
    print(f"{s:10s} {n:55s} {d}")
print(f"killed {sum(s == 'killed' for _, s, _ in res)} / {len(res)}")
