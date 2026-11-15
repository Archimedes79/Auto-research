"""scratch (E343 code step, round 2): each mutation of jtest_eval.py must be caught by tests/test_jtest_eval.py (except the hash test); prints the survivors. usage: mutate.py [--part i/n] [name ...]"""
import os, subprocess, sys
R = os.environ["BFLC_ROOT"]
SC = os.path.join(R, "temp", "e343", "_session_scratch_", "mut")
src = open(os.path.join(R, "evaluation", "jtest_eval.py"), encoding="utf-8").read()
MUT = [
 # reading
 ("reading full D >", 's["D"] >= FULL_D and s["lo"] > 0', 's["D"] > FULL_D and s["lo"] > 0'),
 ("reading full lo >=", 's["D"] >= FULL_D and s["lo"] > 0', 's["D"] >= FULL_D and s["lo"] >= 0'),
 ("reading not lo >=", 's["D"] > 0 and s["lo"] > LO_FLOOR', 's["D"] > 0 and s["lo"] >= LO_FLOOR'),
 ("reading not D >=", 's["D"] > 0 and s["lo"] > LO_FLOOR', 's["D"] >= 0 and s["lo"] > LO_FLOOR'),
 ("reading fallback keep >", 'all(s["D"] >= KEEP_D for s in seeds)', 'all(s["D"] > KEEP_D for s in seeds)'),
 ("reading fallback full >", 'all(s["D"] >= FULL_D for s in seeds):\n        return "FULL RETENTION (without interval)"', 'all(s["D"] > FULL_D for s in seeds):\n        return "FULL RETENTION (without interval)"'),
 ("reading fallback no guards", 'if guards and all(s["D"] >= FULL_D for s in seeds):\n        return "FULL RETENTION (without', 'if all(s["D"] >= FULL_D for s in seeds):\n        return "FULL RETENTION (without'),
 ("reading interval any", 'if all(has(s.get("lo")) for s in seeds):', 'if any(has(s.get("lo")) for s in seeds):'),
 ("guards all", 'guards = all(s["guards"] for s in seeds)', 'guards = any(s["guards"] for s in seeds)'),
 ("reading_inputs lo hi swapped", 'lo=t.get("lo"), hi=t.get("hi")', 'lo=t.get("hi"), hi=t.get("lo")'),
 # write helper, E321
 ("is_protected off", 'return any(real == r or real.startswith(r + os.sep)', 'return False and any(real == r or real.startswith(r + os.sep)'),
 ("is_protected no realpath", 'real = os.path.realpath(path).casefold()', 'real = os.path.abspath(path).casefold()'),
 ("is_protected no casefold", 'real = os.path.realpath(path).casefold()', 'real = os.path.realpath(path)'),
 ("is_protected roots no casefold", '(os.path.realpath(os.path.join(ROOT, p)).casefold() for p in PROTECTED)', '(os.path.realpath(os.path.join(ROOT, p)) for p in PROTECTED)'),
 ("atomic overwrite ignored", 'if not overwrite and os.path.exists(path):', 'if False:'),
 ("e321_changed empty", '    out = []\n    for rel, want in E321_SHA256.items():', '    out = []\n    for rel, want in ():'),
 ("guard_e321 no exit", '    if bad:\n        sys.exit(f"STOP {when}', '    if False:\n        sys.exit(f"STOP {when}'),
 ("e321 missing is unchanged", '        if not os.path.exists(p):\n            out.append(f"{rel} (missing)")', '        if not os.path.exists(p):\n            continue'),
 # gate wiring
 ("cmd_all no gate", 'def cmd_all(a):\n    require_gate(a)\n', 'def cmd_all(a):\n    pass\n'),
 ("main dump no gate", '    require_gate(a)\n    d = dump_dir(a.split, a.max)', '    d = dump_dir(a.split, a.max)'),
 ("b dump made on test", 'b_made=not b_flag and a.split != "test"', 'b_made=not b_flag'),
 # dumps
 ("_require no-op", '    if not cond:\n        raise RuntimeError(msg)', '    if False:\n        raise RuntimeError(msg)'),
 ("validate no rows", '_require([int(d["row"]) for d in items] == rows,', '_require(True or [int(d["row"]) for d in items] == rows,'),
 ("get_dump always make", '    if not os.path.exists(path):\n        if make is None:', '    if True:\n        if make is None:'),
 ("get_dump no provenance", "    elif expect is not None and (make is not None or os.path.exists(head_path(path))):", "    elif False:"),
 ("provenance ignores device", 'for k in ("run", "code_digest", "device", "checkpoint_sha256")', 'for k in ("run", "code_digest", "checkpoint_sha256")'),
 ("provenance ignores code", 'for k in ("run", "code_digest", "device", "checkpoint_sha256")', 'for k in ("run", "device", "checkpoint_sha256")'),
 ("provenance ignores checkpoint", 'for k in ("run", "code_digest", "device", "checkpoint_sha256")', 'for k in ("run", "code_digest", "device")'),
 ("provenance ignores run", 'for k in ("run", "code_digest", "device", "checkpoint_sha256")', 'for k in ("code_digest", "device", "checkpoint_sha256")'),
 ("get_dump flagged unchecked", 'elif expect is not None and (make is not None or os.path.exists(head_path(path))):', 'elif expect is not None and make is not None:'),
 ("get_dump default no sidecar ok", 'elif expect is not None and (make is not None or os.path.exists(head_path(path))):', 'elif expect is not None and os.path.exists(head_path(path)):'),
 ("score_seed B flag unchecked", "\"B's dump\", n, exp(run_b))}", "\"B's dump\", n, exp(run_b) if i[\"b_made\"] else None)}"),
 ("score_seed ctl ends unchecked", "get_dump(i[\"ctl_ends\"], None, split, \"the control's pair dump\", n, exp(sub_seed(CONTROL_RUN, seed)))", "load_dump(i[\"ctl_ends\"], split, \"the control's pair dump\", n)"),
 ("score_seed ctl B unchecked", "get_dump(i[\"ctl_b\"], None, split, \"the control's B dump\", n, exp(run_b))", "load_dump(i[\"ctl_b\"], split, \"the control's B dump\", n)"),
 ("provenance missing sidecar ok", '    if not os.path.exists(hp):\n        sys.exit(', '    if False:\n        sys.exit('),
 ("dump_b overwrites", '    refuse_existing(out)\n    import lane_gt_data as LD\n    import qgraph_readout as RO', '    import lane_gt_data as LD\n    import qgraph_readout as RO'),
 ("dump_ends overwrites", '    refuse_existing(out)\n    import jends_eval as JE', '    import jends_eval as JE'),
 ("head_path no pkl", '_require(out.endswith(".pkl"), ', '_require(True, '),
 ("code skip not followed", 'if not name or name.split(".")[0] in CODE_SKIP:', 'if not name:'),
 ("code no parent packages", ' + [os.path.realpath(c) for c in (os.path.join(d, *parts[:k], "__init__.py") for k in range(1, len(parts))) if os.path.isfile(c)]', ''),
 # control, kill, repro
 ("tol REHEARSAL always", '        tol = tol_of(split)\n', '        tol = REHEARSAL\n'),
 ("tol_of test", 'return EXACT if split == "test" else REHEARSAL', 'return REHEARSAL'),
 ("score_seed no kill", 'if not (ok or n):', 'if False:'),
 ("score_seed counts always", 'control["counts"] = sc["counts"]["e312_snapped"] if control["repro"]["ok"] else None', 'control["counts"] = sc["counts"]["e312_snapped"]'),
 ("score_seed repro tol", 'match(numbers_of(sc["counts"]["e312_snapped"]), view["numbers"], EXACT)', 'match(numbers_of(sc["counts"]["e312_snapped"]), view["numbers"], REHEARSAL)'),
 ("repro ignores wiring", 'ok=bool(r_ok and wiring)', 'ok=bool(r_ok)'),
 ("precheck no seed check", '    check_seed(i["run"], seed)\n    if stack and not pyramid_same', '    if stack and not pyramid_same'),
 ("check_seed no exit", '    if ck.get("seed", seed) != seed:\n        sys.exit(', '    if False:\n        sys.exit('),
 ("precheck no pyramid", '    if stack and not pyramid_same(i["run"], sub_seed(setup["stack_run"], seed)):\n        sys.exit(f"STOP (seed {seed}): the pixel pyramid', '    if False:\n        sys.exit(f"STOP (seed {seed}): the pixel pyramid'),
 ("precheck no missing", '    if missing:\n        sys.exit(f"STOP (seed {seed}): not there', '    if False:\n        sys.exit(f"STOP (seed {seed}): not there'),
 ("cmd_all precheck late", '    pre = {s: precheck(a, setup, s, stack) for s in a.seeds}', '    pre = {s: (None, None) for s in a.seeds}'),
 ("cmd_all results overwrite", '    if old and not a.max:', '    if False:'),
 ("cmd_all drift ignored", 'drift = sorted(f for f in set(code0) | set(code1) if code0.get(f) != code1.get(f))', 'drift = []'),
 ("cmd_all device not in card", 'c.update(device=a.device, ', 'c.update('),
 ("cmd_all rehearsal rc", '    return 0 if ok else 1\n\n\ndef cmd_verify', '    return 0\n\n\ndef cmd_verify'),
 ("cmd_all rehearsal hint", '        if stack and arm_k_only(failed):\n            print("only arm k fails', '        if False:\n            print("only arm k fails'),
 ("cmd_all rehearsal also on smoke", 'rehearsal = a.split == "val" and not a.max', 'rehearsal = a.split == "val"'),
 ("cmd_all use_counts always", 'use_counts = counts_reproduced(done)', 'use_counts = True'),
 ("counts_reproduced any", 'return all(d["control"] and d["control"]["counts"] is not None for d in done)', 'return any(d["control"] and d["control"]["counts"] is not None for d in done)'),
 ("arm_k_only any", 'return bool(failed) and set(failed) <= {"stack"}', 'return bool(failed) and "stack" in set(failed)'),
 ("check_stored looks at unselected", 'if name not in card["variants"]:\n            continue', 'if False:\n            continue'),
 # glue
 ("glue attach dropped", 'pr = pairs_of(ends[v["ends"]][f.row], thr, v["attach"])', 'pr = pairs_of(ends[v["ends"]][f.row], thr, False)'),
 ("glue nogeom dropped", 'snap=v["snap"], snap_nogeom=v["nogeom"])', 'snap=v["snap"], snap_nogeom=False)'),
 ("glue snap dropped", 'snap=v["snap"], snap_nogeom=v["nogeom"])', 'snap=True, snap_nogeom=v["nogeom"])'),
 ("glue graph b", 'graph, bk, long_mask = read[v["graph"]]', 'graph, bk, long_mask = read["b"]'),
 ("glue ends own", 'pr = pairs_of(ends[v["ends"]][f.row]', 'pr = pairs_of(ends["own"][f.row]'),
 # inputs per split
 ("seed_inputs ctl_b ignores flag", 'if cd["b"] and not control_ends_flag else i["b"]', 'if cd["b"] else i["b"]'),
 ("seed_inputs test ctl ends", 'ctl_ends = sub_seed(a.control_ends_dump, seed) or (os.path.join(ROOT, sub_seed(cd["ends"], seed)) if cd["ends"] else None)', 'ctl_ends = sub_seed(a.control_ends_dump, seed) or os.path.join(ROOT, sub_seed(CONTROL_DEFAULTS["val"]["ends"], seed))'),
 ("seed_inputs smoke folder", 'd, run = dump_dir(a.split, a.max),', 'd, run = dump_dir(a.split),'),
 ("out_dir smoke same", '"cards_" + setup["out"]', '"cards"'),
 # cards, guards, match, report
 ("guards dup 0.03", 'dup_frac=g("dup_frac") <= 0.02', 'dup_frac=g("dup_frac") <= 0.03'),
 ("guards f1 0.02", 'F1_1=g("F1@1.0") >= -0.01', 'F1_1=g("F1@1.0") >= -0.02'),
 ("guards vert 2", 'vert100_dp=g("vert100_dp") <= 1.0', 'vert100_dp=g("vert100_dp") <= 2.0'),
 ("guards junction 0.05", 'junction_F1=g("junction_F1") >= -0.03', 'junction_F1=g("junction_F1") >= -0.05'),
 ("guards vs own base", 'guards=guards_vs(num[v_], control["numbers"])', 'guards=guards_vs(num[v_], num[next(x["base"] for x in variants if x["name"] == v_)])'),
 ("arm D from w", 'difference={k: float(num[v_][k]) - control["numbers"][k] for k in KEYS}', 'difference={k: float(num[arms["w"]][k]) - control["numbers"][k] for k in KEYS}'),
 ("card compare uses control counts always", 'if boot and use_counts and control["counts"] is not None:', 'if boot and control["counts"] is not None:'),
 ("results seed 0 only", 'for c, s in zip(cards, seeds):', 'for c, s in zip(cards[:1], seeds[:1]):'),
 ("results reading 4 digits", "{s['D']:+.6f}", "{s['D']:+.4f}"),
 ("match int real tol", 'd <= tol["count_rel"] * abs(v) if isinstance(v, int) else', 'd <= tol["real"] if isinstance(v, int) else'),
 ("match topo real", 'tol["topo" if k == "topo_F1" else "real"]', 'tol["real"]'),
 ("check_stored ignores spec", 'good = o1 and o2 and o3 and not wrong', 'good = o1 and o2 and o3'),
 ("check_stored ignores curves", 'good = o1 and o2 and o3 and not wrong', 'good = o1 and o2 and not wrong'),
 ("counts not enforced", 'good = good and dh == 0.0 and db == 0.0 and same', 'good = good'),
 ("scenes not compared", 'same = st["scenes"] == [f.scene for f in frames] and len(st["head"]) == len(sc["counts"][name])', 'same = True'),
 ("missing counts file passes", '            line += "; per-frame counts: the stored .counts.pkl is MISSING"\n            good = False', '            line += "; per-frame counts: the stored .counts.pkl is MISSING"\n            good = good'),
]
args = sys.argv[1:]
part = None
if args and args[0] == "--part":
    i, n = map(int, args[1].split("/"))
    part = (i, n)
    args = args[2:]
surv = []
for idx, (name, old, new) in enumerate(MUT):
    if part and idx % part[1] != part[0]:
        continue
    if args and name not in args:
        continue
    n = src.count(old)
    if n != 1:
        print(f"MUTATION {name!r}: pattern found {n} times", flush=True)
        surv.append(name + " (pattern)")
        continue
    mod = os.path.join(SC, f"evaluation_{part[0] if part else 0}")
    os.makedirs(mod, exist_ok=True)
    open(os.path.join(mod, "jtest_eval.py"), "w", encoding="utf-8").write(src.replace(old, new))
    env = dict(os.environ, MUT_DIR=mod)
    r = subprocess.run([sys.executable, os.path.join(SC, "driver.py")], capture_output=True, text=True, env=env)
    out = (r.stdout + r.stderr).strip().splitlines()
    last = out[-1] if out else ""
    caught = "FAILED: []" not in last
    print(f"{'caught  ' if caught else 'SURVIVED'} {name}: {last[:150]}", flush=True)
    if not caught:
        surv.append(name)
print("SURVIVORS:", surv)
