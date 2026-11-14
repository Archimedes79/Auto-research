"""scratch (E343 code step): each mutation of jtest_eval.py must be caught by tests/test_jtest_eval.py (except the hash test); prints the survivors"""
import os, shutil, subprocess, sys
R = os.environ["BFLC_ROOT"]
SC = os.path.join(R, "temp", "e343", "_session_scratch_", "mut")
src = open(os.path.join(R, "evaluation", "jtest_eval.py"), encoding="utf-8").read()
MUT = [
 ("reading full D >", 's["D"] >= FULL_D and s["lo"] > 0', 's["D"] > FULL_D and s["lo"] > 0'),
 ("reading full lo >=", 's["D"] >= FULL_D and s["lo"] > 0', 's["D"] >= FULL_D and s["lo"] >= 0'),
 ("reading not lo >=", 's["D"] > 0 and s["lo"] > LO_FLOOR', 's["D"] > 0 and s["lo"] >= LO_FLOOR'),
 ("reading not D >=", 's["D"] > 0 and s["lo"] > LO_FLOOR', 's["D"] >= 0 and s["lo"] > LO_FLOOR'),
 ("reading fallback keep >", 'all(s["D"] >= KEEP_D for s in seeds)', 'all(s["D"] > KEEP_D for s in seeds)'),
 ("reading fallback full >", 'all(s["D"] >= FULL_D for s in seeds):\n        return "FULL RETENTION (without interval)"', 'all(s["D"] > FULL_D for s in seeds):\n        return "FULL RETENTION (without interval)"'),
 ("reading fallback no guards", 'if guards and all(s["D"] >= FULL_D for s in seeds):\n        return "FULL RETENTION (without', 'if all(s["D"] >= FULL_D for s in seeds):\n        return "FULL RETENTION (without'),
 ("reading interval any", 'if all(has(s.get("lo")) for s in seeds):', 'if any(has(s.get("lo")) for s in seeds):'),
 ("guards all", 'guards = all(s["guards"] for s in seeds)', 'guards = any(s["guards"] for s in seeds)'),
 ("is_protected off", 'return any(real == r or real.startswith(r + os.sep)', 'return False and any(real == r or real.startswith(r + os.sep)'),
 ("is_protected no realpath", 'real = os.path.realpath(path)\n    return any', 'real = os.path.abspath(path)\n    return any'),
 ("atomic not atomic", 'os.replace(tmp, path)', 'open(path, "wb").write(open(tmp, "rb").read()); os.remove(tmp)'),
 ("validate no rows", 'assert [int(d["row"]) for d in items] == rows,', 'assert True or [int(d["row"]) for d in items] == rows,'),
 ("validate no len", 'assert len(items) == len(rows),', 'assert True or len(items) == len(rows),'),
 ("get_dump always make", '    if not os.path.exists(path):\n        if make is None:', '    if True:\n        if make is None:'),
 ("glue attach dropped", 'pr = pairs_of(ends[v["ends"]][f.row], thr, v["attach"])', 'pr = pairs_of(ends[v["ends"]][f.row], thr, False)'),
 ("glue nogeom dropped", 'snap=v["snap"], snap_nogeom=v["nogeom"])', 'snap=v["snap"], snap_nogeom=False)'),
 ("glue snap dropped", 'snap=v["snap"], snap_nogeom=v["nogeom"])', 'snap=True, snap_nogeom=v["nogeom"])'),
 ("glue graph b", 'graph, bk, long_mask = read[v["graph"]]', 'graph, bk, long_mask = read["b"]'),
 ("glue ends own", 'pr = pairs_of(ends[v["ends"]][f.row]', 'pr = pairs_of(ends["own"][f.row]'),
 ("score_seed counts always", 'control["counts"] = sc["counts"]["e312_snapped"] if r_ok else None', 'control["counts"] = sc["counts"]["e312_snapped"]'),
 ("score_seed no kill", 'if not (ok or n):', 'if False:'),
 ("score_seed repro tol", 'match(numbers_of(sc["counts"]["e312_snapped"]), control["numbers"], EXACT)', 'match(numbers_of(sc["counts"]["e312_snapped"]), control["numbers"], REHEARSAL)'),
 ("tol_of test", 'return EXACT if split == "test" else REHEARSAL', 'return REHEARSAL'),
 ("guards dup 0.03", 'dup_frac=g("dup_frac") <= 0.02', 'dup_frac=g("dup_frac") <= 0.03'),
 ("guards f1 0.02", 'F1_1=g("F1@1.0") >= -0.01', 'F1_1=g("F1@1.0") >= -0.02'),
 ("guards vert 2", 'vert100_dp=g("vert100_dp") <= 1.0', 'vert100_dp=g("vert100_dp") <= 2.0'),
 ("guards junction 0.05", 'junction_F1=g("junction_F1") >= -0.03', 'junction_F1=g("junction_F1") >= -0.05'),
 ("match int real tol", 'd <= tol["count_rel"] * abs(v) if isinstance(v, int) else', 'd <= tol["real"] if isinstance(v, int) else'),
 ("match topo real", 'tol["topo" if k == "topo_F1" else "real"]', 'tol["real"]'),
 ("check_stored ignores spec", 'good = o1 and o2 and o3 and not wrong', 'good = o1 and o2 and o3'),
 ("check_stored ignores curves", 'good = o1 and o2 and o3 and not wrong', 'good = o1 and o2 and not wrong'),
 ("card compare uses control counts always", 'if boot and use_counts and control["counts"] is not None:', 'if boot and control["counts"] is not None:'),
 ("counts_reproduced any", 'return all(d["control"] and d["control"]["counts"] is not None for d in done)', 'return any(d["control"] and d["control"]["counts"] is not None for d in done)'),
 ("cmd_all use_counts always", 'use_counts = counts_reproduced(done)', 'use_counts = True'),
 ("seed_inputs ctl_b ignores flag", 'if cd["b"] and not control_ends_flag else i["b"]', 'if cd["b"] else i["b"]'),
 ("seed_inputs test ctl ends", 'ctl_ends = sub_seed(a.control_ends_dump, seed) or (os.path.join(ROOT, sub_seed(cd["ends"], seed)) if cd["ends"] else None)', 'ctl_ends = sub_seed(a.control_ends_dump, seed) or os.path.join(ROOT, sub_seed(CONTROL_DEFAULTS["val"]["ends"], seed))'),
 ("seed_inputs smoke folder", 'd, run = dump_dir(a.split, a.max),', 'd, run = dump_dir(a.split),'),
 ("require_gate max", 'if getattr(a, "max", 0):', 'if False:'),
 ("b dump made on test", 'b_made=not b_flag and a.split != "test"', 'b_made=not b_flag'),
]
only = sys.argv[1:]
skipping = bool(only and only[0] == "--from")
surv = []
for name, old, new in MUT:
    if skipping and name == only[1]:
        skipping = False
    if skipping or (only and only[0] != "--from" and name not in only):
        continue
    n = src.count(old)
    if n != 1:
        print(f"MUTATION {name!r}: pattern found {n} times", flush=True)
        surv.append(name + " (pattern)")
        continue
    open(os.path.join(SC, "evaluation", "jtest_eval.py"), "w", encoding="utf-8").write(src.replace(old, new))
    r = subprocess.run([sys.executable, os.path.join(SC, "driver.py")], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    last = out[-1] if out else ""
    caught = "FAILED: []" not in last
    print(f"{'caught  ' if caught else 'SURVIVED'} {name}: {last[:160]}", flush=True)
    if not caught:
        surv.append(name)
print("SURVIVORS:", surv)
