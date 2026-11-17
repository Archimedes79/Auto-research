def check_seed(run, seed):
    """the pair run is the run of this seed (its keypoints sit on that seed's graph): its `config.json` says so"""
    with open(os.path.join(ROOT, "runs", run, "config.json")) as f:
        ck = json.load(f)
    if ck.get("seed", seed) != seed:
        sys.exit(f"the run {run} has seed {ck.get('seed')}, not {seed}: its keypoints would sit on another seed's graph")


def precheck(a, setup, seed, stack):
    """everything that can be refused before hours of dumping: the seed of the pair run, the pixel pyramid of the stack (arm k fails its own check: run again with --no-stack), the dumps that a flag or the split's defaults name, the control's card
    -> (the paths of the seed, E312's card or None)"""
    i = seed_inputs(a, setup, seed, stack, bool(a.control_ends_dump))
    check_seed(i["run"], seed)
    if stack and not pyramid_same(i["run"], sub_seed(setup["stack_run"], seed)):
        sys.exit(f"STOP (seed {seed}): the pixel pyramid of {i['run']} is not that of {sub_seed(setup['stack_run'], seed)}, so the stack's connectors would sit on other pixel features than the delivered graph: arm k fails its check, run again with --no-stack")
    named = ([] if i["b_made"] else [i["b"]]) + ([i["ctl_ends"], i["ctl_b"]] if setup["arms"] and i["ctl_ends"] else []) + ([i["card"]] if setup["arms"] else [])
    missing = [p for p in named if not os.path.exists(p)]
    if missing:
        sys.exit(f"STOP (seed {seed}): not there, and named by a flag or by the split's defaults (B's dump is never made on TEST): {missing}")
    ctl = None
    if setup["arms"]:
        with open(i["card"]) as f:
            ctl = json.load(f)
    return i, ctl


def score_seed(a, setup, seed, frames, stack, pre):
    """one seed (`pre`: what `precheck` returned): the dumps (validated; one that this module made earlier is reused only under the same provenance), the scoring of every variant, the control's checks
    -> dict(seed, run, sc, variants, control); stops before any number is printed when the plain graph differs from the control card's"""
    i, ctl_card = pre
    n, split = a.max, a.split
    run_b, run_c = sub_seed(setup["graph_run"], seed), sub_seed(setup.get("stack_run"), seed)
    exp = lambda run: provenance(run, a.device)  # noqa: E731
    dumps = {"b": get_dump(i["b"], (lambda: dump_b(split, run_b, i["b"], a.device, n)) if i["b_made"] else None, split, "B's dump", n, exp(run_b) if i["b_made"] else None)}
    ends = {"own": {f["row"]: f for f in get_dump(i["ends"], lambda: dump_ends(split, i["run"], i["ends"], a.device, n), split, f"pair dump of {i['run']}", n, exp(i["run"]))}}
    if stack:
        dumps["c2"] = get_dump(i["c2"], lambda: dump_b(split, run_c, i["c2"], a.device, n), split, "c2's dump", n, exp(run_c))
    control = view = None
    if setup["arms"]:
        view = control_view(ctl_card)
        control = dict(card=os.path.relpath(i["card"], ROOT).replace(os.sep, "/"), numbers=view["numbers"], counts=None, repro=None)
        if i["ctl_ends"]:
            dumps["ctl"] = dumps["b"] if i["ctl_b"] == i["b"] else load_dump(i["ctl_b"], split, "the control's B dump", n)
            ends["ctl"] = {f["row"]: f for f in load_dump(i["ctl_ends"], split, "the control's pair dump", n)}
    variants = variants_of(setup, stack, control=bool(control) and "ctl" in ends)
    sc = score_dumps(frames, dumps, ends, variants, n=n)
    if control:
        tol = tol_of(split)
        ok, worst, where = match(numbers_of(sc["counts"]["plain"]), view["plain"], tol)
        control["plain_check"] = dict(max_abs_diff=worst, where=where, tolerance=f"topo {tol['topo']:g}, others {tol['real']:g}", ok=bool(ok or n))                  # a smoke run (the first n frames) cannot reproduce a whole card
        if not (ok or n):
            sys.exit(f"KILL (seed {seed}): the plain graph computed from B's dump differs from the control card's numbers.base by {worst:.2e} ({where}): no number of any arm is used")
        if "ctl" in ends:
            r_ok, r_worst, r_where = match(numbers_of(sc["counts"]["e312_snapped"]), view["numbers"], EXACT)
            wiring = (sc["curves"]["e312_snapped"], sc["wired"]["e312_snapped"]) == (view["curves"], view["wired"])                                                 # E312's curves and wired counts too
            control["repro"] = dict(max_abs_diff=r_worst, where=r_where, ok=bool(r_ok and wiring), curves=[sc["curves"]["e312_snapped"], view["curves"]], wired=[sc["wired"]["e312_snapped"], view["wired"]])
            control["counts"] = sc["counts"]["e312_snapped"] if control["repro"]["ok"] else None
    return dict(seed=seed, run=i["run"], sc=sc, variants=variants, control=control)


def check_stored(setup, run, seed, card, sc, tol, frames=None):
    """the stored VAL cards of the setup against the card just made: the numbers (head, plain, difference), curves and wired counts within `tol`, the stored spec fields equal to the variant's; with EXACT also the per-frame counts equal to the stored
    `.counts.pkl` and the scenes equal (a card of `setup["stored_counts"]` without that file fails) -> (ok, lines, failed): `failed` names the read-outs whose stored card did not reproduce. Only the read-outs in `card` are checked: without the stack
    (`--no-stack`) the stack card is not looked at and w and p are checked alone"""
    lines, failed = [], []
    by_name = {v["name"]: v for v in setup["variants"] + setup["stack"]}
    for name, fname in setup["stored"].items():
        if name not in card["variants"]:
            continue
        path = os.path.join(ROOT, "runs", run, fname)
        old, new, v = json.load(open(path)), card["variants"][name], by_name[name]
        want = dict(thr=THR, spec=CLOSE, snap=v["snap"], attach=v["attach"], attach_r=None, snap_options=["snap_nogeom"] if v["nogeom"] else [], wire=None)
        if tol is EXACT:                                                                              # the stored dumps made the stored cards; a fresh dump has another path
            want["full"] = sub_seed(STORED_DUMPS[v["graph"]], seed)
        wrong = {k: (old[k], w) for k, w in want.items() if k in old and old[k] != w}
        o1, w1, wh1 = match(new["numbers"], old["numbers"], tol)
        o2, w2, wh2 = match(new["difference"], old["difference"], tol)
        o3, w3, wh3 = match(dict(curves=new["curves"], wired=new["wired_both_ends"]), dict(curves=old["curves"], wired=old["wired_both_ends"]), tol)
        line = f"CHECK {name}: max abs difference of the numbers {w1:.2e} ({wh1}), of the differences {w2:.2e}; curves {new['curves']} / {old['curves']}, wired {new['wired_both_ends']} / {old['wired_both_ends']} ({'within' if o3 else 'OUT'})"
        good = o1 and o2 and o3 and not wrong
        if wrong:
            line += f"; STORED SPEC DIFFERS {wrong}"
        cpath = path[:-5] + ".counts.pkl"
        if tol is EXACT and os.path.exists(cpath):
            with open(cpath, "rb") as f:
                st = pickle.load(f)
            dh, db = counts_maxdiff(sc["counts"][name], st["head"]), counts_maxdiff(sc["counts"][v["base"]], st["base"])
            same = st["scenes"] == [f.scene for f in frames] and len(st["head"]) == len(sc["counts"][name])
            line += f"; per-frame counts: max abs difference {max(dh, db):.2e} (head {dh:.2e}, plain {db:.2e}) over {len(st['head'])} frames, scenes {'equal' if same else 'DIFFER'}"
            good = good and dh == 0.0 and db == 0.0 and same
        elif tol is EXACT and name in setup["stored_counts"]:
            line += "; per-frame counts: the stored .counts.pkl is MISSING"
            good = False
        elif tol is EXACT:
            line += "; per-frame counts: no stored file"
        if not good:
            failed.append(name)
        lines.append(line + ("" if good else "  <-- DIFFERENCE"))
    if setup["stored_man"]:
        with open(os.path.join(ROOT, "runs", run, setup["stored_man"])) as f:
            old = json.load(f)["variants"]
        for lab, new in card["maneuver"].items():
            o, w, wh = match(new, old[lab], tol)
            if not o:
                failed.append(f"maneuver {lab}")
            lines.append(f"CHECK maneuver {lab}: realised {new['realised']} / {old[lab]['realised']} of {new['maneuvers']} / {old[lab]['maneuvers']}, max abs difference {w:.2e} ({wh})" + ("" if o else "  <-- DIFFERENCE"))
    else:
        lines.append("maneuver coverage: not verified against a stored card (none exists for this setup): " + ", ".join(f"{k} {100 * m['recall']:.1f} %" for k, m in card["maneuver"].items()))
    return not failed, lines, failed


def arm_k_only(failed):
    """does ONLY the stack (arm k) fail its check? Then the card's rule applies: run again with --no-stack, w and p are checked alone (decided from VAL only)"""
    return bool(failed) and set(failed) <= {"stack"}


def counts_reproduced(done):
    """E312's per-frame counts give the paired intervals only when they reproduced the stored card on EVERY seed (else no interval on any seed: the fallback of `reading`)"""
    return all(d["control"] and d["control"]["counts"] is not None for d in done)


def cmd_all(a):
    require_gate(a)
    setup = SETUPS.get(a.run_prefix) or sys.exit(f"--run-prefix {a.run_prefix!r}: one of {', '.join(SETUPS)}")
    stack = bool(setup["stack"]) and not a.no_stack
    out = out_dir(setup, a.split, a.max)
    for path in (out, dump_dir(a.split, a.max)):                                                     # before hours of work: E321's folders cannot be written (E312 on TEST is E321's, already done)
        if is_protected(path):
            sys.exit(f"refused: {path} lies under {' / '.join(PROTECTED)}")
    targets = [os.path.join(out, f"card_s{s}.json") for s in a.seeds] + [os.path.join(out, "results.md")]
    old = [p for p in targets if os.path.exists(p)]
    if old and not a.max:
        sys.exit(f"refused: {old} exist: the cards and results.md are written once (move them away to run again)")
    guard_e321("before the run")
    pre = {s: precheck(a, setup, s, stack) for s in a.seeds}                                         # seeds, pyramid, flags, control card: refused here, not after hours of dumping
    frames = frames_of(a.split)[: a.max or None]
    code0, done = code_sha256(), []
    for s in a.seeds:
        t0 = time.time()
        done.append(score_seed(a, setup, s, frames, stack, pre[s]))
        print(f"seed {s} scored in {time.time() - t0:.0f} s", flush=True)
    use_counts = counts_reproduced(done)
    if setup["arms"] and not use_counts:
        print("the per-frame counts of E312 are not used or did not reproduce on every seed: NO interval against E312 on any seed (the fallback of reading())", flush=True)
    rehearsal = a.split == "val" and not a.max
    cards, ok, failed = [], True, []
    for d in done:
        card = make_card(d["seed"], d["run"], a.split, frames[: len(d["sc"]["counts"]["plain"])], d["sc"], a.boot, setup, d["variants"], d["control"], use_counts)
        if rehearsal:                                                                                # the dress rehearsal: the fresh dumps against the stored VAL cards, the tolerance of criterion 2
            good, lines, bad = check_stored(setup, d["run"], d["seed"], card, d["sc"], REHEARSAL)
            card["rehearsal_check"] = dict(ok=bool(good), lines=lines, failed=bad)
            ok, failed = ok and good, failed + bad
        cards.append(card)
    code1 = code_sha256()                                                                            # the imported lane code at the end: a file that changed during the run is named (not fatal)
    drift = sorted(f for f in set(code0) | set(code1) if code0.get(f) != code1.get(f))
    for c in cards:
        c.update(device=a.device, code_sha256=code0, code_digest=code_digest(code0), code_drift=drift)
    if drift:
        print("CODE DRIFT:", ", ".join(drift), flush=True)
    for c in cards:
        print_card(c)
        for line in c.get("rehearsal_check", {}).get("lines", []):
            print(line)
    for c in cards:
        save_json(os.path.join(out, f"card_s{c['seed']}.json"), c, overwrite=bool(a.max))
    save_text(os.path.join(out, "results.md"), results_md(cards, a.split, setup), overwrite=bool(a.max))
    if setup["arms"] and len(cards) == 2:
        print("READING:", reading([reading_inputs(c) for c in cards]))
    guard_e321("after the run")
    if rehearsal:
        print("REHEARSAL CHECK:", "PASS" if ok else "FAIL (a tolerance of criterion 2 is exceeded)")
        if stack and arm_k_only(failed):
            print("only arm k fails its check: run again with --no-stack (w and p are then checked alone)")
    return 0 if ok else 1


def cmd_verify(a):
    """VAL, from the stored dumps: the scorer of this module against the stored cards of the setup (criterion 1 of E343: every number within 1e-9, curves and wired counts equal, per-frame counts equal to the stored `.counts.pkl`). Without `--no-stack` the stack
    (arm k) is checked too and its failure makes the exit code non-zero; with it w and p are checked alone"""
    run = a.run or f"e312_p_s{a.seed}"
    setup, seed = setup_of(run)
    stack = bool(setup["stack"]) and not a.no_stack
    variants = variants_of(setup, stack)
    frames = frames_of("val")
    dumps = {"b": load_dump(os.path.join(ROOT, sub_seed(STORED_DUMPS["b"], seed)), "val", "the stored B dump")}
    if stack:
        dumps["c2"] = load_dump(os.path.join(ROOT, sub_seed(STORED_DUMPS["c2"], seed)), "val", "the stored c2 dump")
    ends = {"own": {f["row"]: f for f in load_dump(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "val", f"the stored pair dump of {run}")}}
    check_seed(run, seed)
    sc = score_dumps(frames, dumps, ends, variants)
    card = make_card(seed, run, "val", frames, sc, a.boot, setup, variants)
    save_json(os.path.join(dump_dir("val"), "val_check", f"verify_card_{run}.json"), card)                # for E312 in the layout of E321's cards: the control card of a rehearsal
    print_card(card)
    ok, lines, failed = check_stored(setup, run, seed, card, sc, EXACT, frames)
    for line in lines:
        print("VERIFY " + line)
    if stack:                                                                                          # the stored stack card was made under this check of `jends_eval.py score --base`
        same = pyramid_same(run, sub_seed(setup["stack_run"], seed))
        print(f"VERIFY pyramid of {run} == the pyramid of {sub_seed(setup['stack_run'], seed)}: {same}")
        if not same:
            ok, failed = False, failed + ["stack"]
    print("VERIFY RESULT:", f"all reproduced ({run}: numbers to 1e-9, curves and wired counts equal, per-frame counts equal)" if ok else "DIFFERENCES (see above)")
    if stack and arm_k_only(failed):
        print("VERIFY only arm k fails its check: run again with --no-stack (w and p are then checked alone)")
    return 0 if ok else 1


