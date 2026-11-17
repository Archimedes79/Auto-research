"""scratch (E343 code step, round 4): the critic's findings applied to evaluation/jtest_eval.py (each old text must be found once)"""
import os

P = os.path.join(os.environ["BFLC_ROOT"], "evaluation", "jtest_eval.py")
s = open(P, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------------------------------------------- the refusal message of a dump without provenance (the sidecar is written first now)
rep('''sys.exit(f"{what}: {path} has no provenance: made by other code OR interrupted between the pickle and its sidecar {head_path(path)}: move it away and run again")''',
    '''sys.exit(f"{what}: {path} has no provenance (no sidecar {head_path(path)} with a provenance key): made by other code or by an older version of this module: move it away and run again")''')

# ---------------------------------------------------------------------------------------------------- dump_b / dump_ends: GPU gate, sidecar first, pickle never overwritten
rep('''    prov = provenance(run, device)                                                                     # the code and the checkpoint as they are when the dump STARTS (not after 10 - 25 minutes)
    import lane_gt_data as LD''', '''    prov = provenance(run, device)                                                                     # the code and the checkpoint as they are when the dump STARTS (not after 10 - 25 minutes)
    if device != "cpu":                                                                                # D100: a dump next to the trainings waits for room on the GPU, as `jends_eval.py dump` does
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtest_eval dump {run}")
    write_head(out, dict(split=split, frames=None), prov)                                              # the sidecar FIRST: a crash leaves a sidecar without a pickle, never a pickle without provenance
    import lane_gt_data as LD''')
rep('''    with atomic(out) as tmp:
        RO.dump_full(preds, rows[: len(preds)], tmp)
    write_head(out, dict(split=split, frames=len(preds)), prov)''', '''    with atomic(out, overwrite=False) as tmp:
        RO.dump_full(preds, rows[: len(preds)], tmp)
    write_head(out, dict(split=split, frames=len(preds)), prov)''')
rep('''    prov = provenance(run, device)                                                                     # at the start, as in dump_b
    import jends_eval as JE''', '''    prov = provenance(run, device)                                                                     # at the start, as in dump_b
    if device != "cpu":
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jtest_eval dump {run}")
    write_head(out, dict(run=run, split=split, frames=None), prov)                                     # the sidecar first, as in dump_b
    import jends_eval as JE''')
rep('''    save_pickle(out, out_frames)
    write_head(out, head, prov)''', '''    save_pickle(out, out_frames, overwrite=False)
    write_head(out, head, prov)''')

# ---------------------------------------------------------------------------------------------------- check_seed: the seed must be there
a = s.index("def check_seed(run, seed):")
b = s.index("def precheck(a, setup, seed, stack):")
s = s[:a] + '''def check_seed(run, seed):
    """the pair run is the run of this seed (its keypoints sit on that seed's graph): its `config.json` HAS a seed and it is this one. B (`e273_p_s<S>`) and c2 (`e292_c2_s<S>`) have no config.json: their seed is the `_s<S>` of their name
    (`--b-run` checks it for the dumps) and, for c2, the pixel pyramid equal to the pair run's (`pyramid_same`)"""
    with open(os.path.join(ROOT, "runs", run, "config.json")) as f:
        ck = json.load(f)
    _require("seed" in ck, f"runs/{run}/config.json has no seed: the run cannot be tied to the seed {seed}")
    if ck["seed"] != seed:
        sys.exit(f"the run {run} has seed {ck['seed']}, not {seed}: its keypoints would sit on another seed's graph")


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- precheck: E321's hashed artefacts on TEST, the rows of every existing dump
rep('''    for path, run, what, required in dumps:                                                            # a dump that exists under other provenance is refused here, before any dump of any seed is made
        if os.path.exists(path) and (required or sidecar_provenance(path) is not None):
            check_provenance(path, provenance(run, a.device), what, required)''', '''    for path, run, what, required in dumps:                                                            # a dump that exists under other provenance, or with other rows, is refused here, before any dump of any seed is made
        if not os.path.exists(path):
            continue
        if required or sidecar_provenance(path) is not None:
            check_provenance(path, provenance(run, a.device), what, required)
        elif a.split == "test":                                                                        # on TEST a dump of another origin must be one of E321's hashed artefacts (guard_e321 hashes them)
            rel = os.path.relpath(os.path.realpath(path), os.path.realpath(ROOT)).replace(os.sep, "/")
            _require(rel in E321_SHA256, f"{what}: {path} has no provenance and is not one of E321's hashed artefacts ({', '.join(E321_SHA256)})")
        load_dump(path, a.split, what, a.max)                                                          # its rows too (validated and let go: loaded again when it is used)''')

# ---------------------------------------------------------------------------------------------------- score_seed, the control's checks, the early pass
a = s.index("def score_seed(a, setup, seed, frames, stack, pre):")
b = s.index("def check_stored(setup, run, seed, card, sc, tol, frames=None):")
s = s[:a] + '''def new_control(i, view):
    return dict(card=os.path.relpath(i["card"], ROOT).replace(os.sep, "/"), numbers=view["numbers"], counts=None, repro=None)


def control_checks(split, n, seed, control, view, plain_counts, ctl_counts=None, curves=None, wired=None):
    """the two checks of E312's control on one seed: the plain graph computed from B's dump equals the control card's (KILL: no number of any arm is used) and, when E312's pair head was re-scored, E312's snapped numbers, curves and wired counts equal the stored
    card's to 1e-9 (then its per-frame counts give the intervals) -> `control`, filled"""
    tol = tol_of(split)
    ok, worst, where = match(numbers_of(plain_counts), view["plain"], tol)
    control["plain_check"] = dict(max_abs_diff=worst, where=where, tolerance=f"topo {tol['topo']:g}, others {tol['real']:g}", ok=bool(ok or n))                  # a smoke run (the first n frames) cannot reproduce a whole card
    if not (ok or n):
        sys.exit(f"KILL (seed {seed}): the plain graph computed from B's dump differs from the control card's numbers.base by {worst:.2e} ({where}): no number of any arm is used")
    if ctl_counts is not None:
        r_ok, r_worst, r_where = match(numbers_of(ctl_counts), view["numbers"], EXACT)
        wiring = (curves, wired) == (view["curves"], view["wired"])                                                                                           # E312's curves and wired counts too
        control["repro"] = dict(max_abs_diff=r_worst, where=r_where, ok=bool(r_ok and wiring), curves=[curves, view["curves"]], wired=[wired, view["wired"]])
        control["counts"] = ctl_counts if control["repro"]["ok"] else None
    return control


def control_early(a, setup, seed, frames, pre):
    """the control's two checks BEFORE any new dump is made (a KILL after hours must not be possible): when B's dump and E312's pair dump are there already (named by flags: on TEST E321's), score only the plain graph and E312's snapped read-out from them and
    run `control_checks`; `score_seed` reuses these counts and scores the other read-outs only -> None (nothing to do early: B's dump is still to be made) or dict(sc, control)"""
    i, ctl_card = pre
    if not (setup["arms"] and i["ctl_ends"] and not i["b_made"]):
        return None
    n, split = a.max, a.split
    run_b = sub_seed(setup["graph_run"], seed)
    exp = lambda run: provenance(run, a.device)  # noqa: E731
    dumps = {"b": get_dump(i["b"], None, split, "B's dump", n, exp(run_b))}
    dumps["ctl"] = dumps["b"] if i["ctl_b"] == i["b"] else get_dump(i["ctl_b"], None, split, "the control's B dump", n, exp(run_b))
    ends = {"ctl": {f["row"]: f for f in get_dump(i["ctl_ends"], None, split, "the control's pair dump", n, exp(sub_seed(CONTROL_RUN, seed)))}}
    sc = score_dumps(frames, dumps, ends, [v for v in setup["variants"] if v["name"] == "plain"] + [CONTROL_VARIANT], n=n)
    view = control_view(ctl_card)
    return dict(sc=sc, control=control_checks(split, n, seed, new_control(i, view), view, sc["counts"]["plain"], sc["counts"]["e312_snapped"], sc["curves"]["e312_snapped"], sc["wired"]["e312_snapped"]))


def score_seed(a, setup, seed, frames, stack, pre, early=None):
    """one seed (`pre`: what `precheck` returned, `early`: what `control_early` returned): the dumps (validated; one that this module made earlier is reused only under the same provenance), the scoring of every variant (not those of `early`: their counts are
    taken over), the control's checks -> dict(seed, run, sc, variants, control); stops before any number is printed when the plain graph differs from the control card's"""
    i, ctl_card = pre
    n, split = a.max, a.split
    run_b, run_c = sub_seed(setup["graph_run"], seed), sub_seed(setup.get("stack_run"), seed)
    exp = lambda run: provenance(run, a.device)  # noqa: E731
    dumps = {"b": get_dump(i["b"], (lambda: dump_b(split, run_b, i["b"], a.device, n)) if i["b_made"] else None, split, "B's dump", n, exp(run_b))}
    ends = {"own": {f["row"]: f for f in get_dump(i["ends"], lambda: dump_ends(split, i["run"], i["ends"], a.device, n), split, f"pair dump of {i['run']}", n, exp(i["run"]))}}
    if stack:
        dumps["c2"] = get_dump(i["c2"], lambda: dump_b(split, run_c, i["c2"], a.device, n), split, "c2's dump", n, exp(run_c))
    control = view = None
    if setup["arms"]:
        view = control_view(ctl_card)
        control = early["control"] if early else new_control(i, view)
        if i["ctl_ends"] and not early:
            dumps["ctl"] = dumps["b"] if i["ctl_b"] == i["b"] else get_dump(i["ctl_b"], None, split, "the control's B dump", n, exp(run_b))
            ends["ctl"] = {f["row"]: f for f in get_dump(i["ctl_ends"], None, split, "the control's pair dump", n, exp(sub_seed(CONTROL_RUN, seed)))}
    variants = variants_of(setup, stack, control=bool(control) and bool(i["ctl_ends"]))
    sc = score_dumps(frames, dumps, ends, [v for v in variants if not (early and v["name"] in early["sc"]["counts"])], n=n)
    if early:
        sc = {k: {**early["sc"][k], **sc[k]} for k in sc}
    elif control:
        control_checks(split, n, seed, control, view, sc["counts"]["plain"], sc["counts"]["e312_snapped"] if "ctl" in ends else None, sc["curves"].get("e312_snapped"), sc["wired"].get("e312_snapped"))
    return dict(seed=seed, run=i["run"], sc=sc, variants=variants, control=control)


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- cmd_all with the code pin, the environment, the early control pass
a = s.index("def cmd_all(a):")
b = s.index("def cmd_verify(a):")
s = s[:a] + '''def validate_rehearsal_card(card, run_prefix):
    """a card may pin the code of the TEST run only when it is the card of a complete, passed and undisturbed VAL rehearsal of this system: a full VAL card (not a TEST card, not a smoke `--max` card), the rehearsal check passed, no lane code file changed during
    the run (the card stores the digest of the START state: a run during which a file changed does not name the code that ran), its digest is the digest of its own file list, and it records the environment -> the card"""
    _require(isinstance(card, dict) and card.get("split") == "val" and card.get("frames") == EXPECT_FRAMES["val"], "--expect-code-card: not the card of a full VAL rehearsal (a TEST card or a smoke card)")
    _require(str(card.get("run", "")).startswith(run_prefix), f"--expect-code-card: the card is of the run {card.get('run')!r}, not of {run_prefix}<seed>")
    rc = card.get("rehearsal_check")
    _require(isinstance(rc, dict) and rc.get("ok") is True, "--expect-code-card: the rehearsal check of that card did not pass (or it has none: a smoke card)")
    _require("code_drift" in card and not card["code_drift"], f"--expect-code-card: lane code changed during that rehearsal ({card.get('code_drift')}): its digest does not name the code that ran")
    files = card.get("code_sha256")
    _require(isinstance(files, dict) and bool(files) and code_digest(files) == card.get("code_digest"), "--expect-code-card: the card's code digest is not the digest of its own file list")
    _require(isinstance(card.get("environment"), dict), "--expect-code-card: the card has no environment record")
    return card


def pin_code(a, run_prefix, code0, digest):
    """the code pin: the digest of the code that runs now must be the rehearsal card's (`--expect-code-card`, validated; REQUIRED on TEST: the expected digest comes from the card, no second typed value) and, when given, `--expect-code-digest` (an optional extra
    check: the full digest or its first 16 characters; an empty string refuses, it does not skip) -> the card or None; exits when the code differs"""
    ref = None
    if a.expect_code_card:
        with open(a.expect_code_card) as f:
            ref = validate_rehearsal_card(json.load(f), run_prefix)

    def changed():
        if not ref:
            return "; give --expect-code-card <the rehearsal card> to name the files"
        ch = sorted(k for k in set(ref["code_sha256"]) | set(code0) if ref["code_sha256"].get(k) != code0.get(k))
        return f"; changed files against {a.expect_code_card}: {', '.join(ch[:12])}{' ...' if len(ch) > 12 else ''}" if ch else "; the card's file hashes equal the current ones: that digest is not this card's"
    want = a.expect_code_digest
    if want is not None and not (digest == want or (len(want) >= 16 and digest.startswith(want))):
        sys.exit(f"STOP: the imported lane code (digest {digest}) is not the code of the rehearsal (--expect-code-digest {want!r}){changed()}")
    if ref and ref["code_digest"] != digest:
        sys.exit(f"STOP: the imported lane code (digest {digest}) is not the code of the rehearsal (the digest of --expect-code-card is {ref['code_digest']}){changed()}")
    return ref


def environment(setup, run_prefix, seeds):
    """the small files and library versions the numbers depend on, for the cards (on TEST compared with the rehearsal's card): the sha256 of data/splits.json, val_tokens.json and `runs/<run>/config.json` of every run of the system (B and c2 have no
    config.json: absent is a value too), the versions of torch, numpy and shapely (the big cache files are not hashed)"""
    import importlib
    import qgraph_eval as QE
    paths = {"data/splits.json": os.path.join(ROOT, "data", "splits.json"), "val_tokens.json": os.path.join(QE.LS.B123, "feats", "val_tokens.json")}
    for sd in seeds:
        for run in [sub_seed(setup["graph_run"], sd), f"{run_prefix}{sd}"] + ([sub_seed(setup["stack_run"], sd)] if setup.get("stack_run") else []) + ([sub_seed(CONTROL_RUN, sd)] if setup["arms"] else []):
            paths[f"runs/{run}/config.json"] = os.path.join(ROOT, "runs", run, "config.json")
    versions = {}
    for mod in ("torch", "numpy", "shapely"):
        try:
            versions[mod] = str(importlib.import_module(mod).__version__)
        except Exception:
            versions[mod] = None
    return dict(files={k: (sha256_file(p) if os.path.isfile(p) else None) for k, p in sorted(paths.items())}, versions=versions)


def environment_diff(now, ref):
    return sorted(f"{sec}/{k}" for sec in ("files", "versions") for k in set(now.get(sec, {})) | set(ref.get(sec, {})) if now.get(sec, {}).get(k) != ref.get(sec, {}).get(k))


def repo_modules():
    """the loaded modules whose files lie in the repository (not under temp/, not scratch, not GtGenerator), as paths from the repository root"""
    repo = os.path.realpath(os.path.join(ROOT, ".."))
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f or not os.path.isabs(f) or not os.path.realpath(f).startswith(repo + os.sep):
            continue
        rel = os.path.relpath(os.path.realpath(f), repo).replace(os.sep, "/")
        if rel.split("/")[0] != "GtGenerator" and "/temp/" not in "/" + rel and "_session_scratch_" not in rel:
            out.add(rel)
    return out


def cmd_all(a):
    require_gate(a)
    setup = SETUPS.get(a.run_prefix) or sys.exit(f"--run-prefix {a.run_prefix!r}: one of {', '.join(SETUPS)}")
    stack = bool(setup["stack"]) and not a.no_stack
    if a.split == "test":                                                                              # the card fixes these: 1000 draws, seeds 0 1, E321's dumps and card as the control, the rehearsal's card as the code pin
        _require(a.boot == 1000, "TEST: the card fixes 1000 bootstrap draws (--boot 1000)")
        _require(list(a.seeds) == [0, 1], "TEST: the card fixes the seeds 0 1")
        _require(bool(a.b_dump) and bool(a.control_ends_dump) and bool(a.control_card), "TEST: --b-dump, --control-ends-dump and --control-card name the control inputs (E321's dumps and card)")
        _require(bool(a.expect_code_card), "TEST: --expect-code-card is required (the card of the VAL rehearsal that pins the code: the expected digest is taken from it)")
    out = out_dir(setup, a.split, a.max)
    for path in (out, dump_dir(a.split, a.max)):                                                     # before hours of work: E321's folders cannot be written (E312 on TEST is E321's, already done)
        if is_protected(path):
            sys.exit(f"refused: {path} lies under {' / '.join(PROTECTED)}")
    targets = [os.path.join(out, f"card_s{s}.json") for s in a.seeds] + [os.path.join(out, "results.md")]
    old = [p for p in targets if os.path.exists(p)]
    if old and not a.max:
        sys.exit(f"refused: {old} exist: the cards and results.md are written once (move them away to run again)")
    loaded0 = repo_modules()                                                                          # what the gate and the start loaded: not a module of this run's own imports
    code0 = code_sha256()
    digest = code_digest(code0)
    print(f"code digest: {digest} ({len(code0)} files)", flush=True)
    ref = pin_code(a, a.run_prefix, code0, digest)
    env0 = environment(setup, a.run_prefix, a.seeds)
    if ref:
        diff = environment_diff(env0, ref["environment"])
        if diff:
            sys.exit(f"STOP: the environment differs from the rehearsal's (--expect-code-card {a.expect_code_card}): {', '.join(diff[:12])}")
    guard_e321("before the run")
    pre = {s: precheck(a, setup, s, stack) for s in a.seeds}                                         # seeds, pyramid, flags, control card, provenance and rows of the existing dumps: refused here, not after hours of dumping
    frames = frames_of(a.split)[: a.max or None]
    if a.split == "test":
        _require(len({f.scene for f in frames}) == 21, "TEST: the frames must have 21 scenes (a degenerate scene column would collapse the paired interval to a point)")
    early = {s: control_early(a, setup, s, frames, pre[s]) for s in a.seeds}                         # the control's two checks before the first new dump
    done = []
    for s in a.seeds:
        t0 = time.time()
        done.append(score_seed(a, setup, s, frames, stack, pre[s], early[s]))
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
    undigested = sorted(repo_modules() - loaded0 - set(code0))                                       # a repository module this run loaded that the digest does not cover: named (not fatal; the rehearsal shows it)
    for c in cards:
        c.update(device=a.device, code_sha256=code0, code_digest=code_digest(code0), code_drift=drift, environment=env0, undigested_modules=undigested)
    if drift:
        print("CODE DRIFT:", ", ".join(drift), flush=True)
    for m in undigested:
        print("UNDIGESTED MODULE:", m, flush=True)
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


''' + s[b:]

# ---------------------------------------------------------------------------------------------------- results.md: undigested modules, environment
rep('''(f"CODE DRIFT during the run: {', '.join(drift)}." if drift else "no change of those files during the run.")]''',
    '''(f"CODE DRIFT during the run: {', '.join(drift)}." if drift else "no change of those files during the run.")]
        L += [f"UNDIGESTED MODULE: {m}" for m in c0.get("undigested_modules") or []]
        if c0.get("environment"):
            L += ["", "Environment: " + ", ".join(f"{k} {v}" for k, v in c0["environment"]["versions"].items()) + f"; {len(c0['environment']['files'])} small files hashed (data/splits.json, val_tokens.json, run configs)."]''')

# ---------------------------------------------------------------------------------------------------- docstring
rep('''Before any dump is made, `all` checks the seeds of the pair runs, the pixel pyramid of the stack, the dumps named by flags and the control card (refused there, not after hours).''',
    '''Before any dump is made, `all` checks the seeds of the pair runs (`config.json` must have one), the pixel pyramid of the stack, the dumps named by flags and the control card, and the provenance AND rows of every dump that exists
(a dump of another origin on TEST must be one of E321's hashed artefacts); it then scores E312's control from the flagged inputs (E321's dumps on TEST): the plain graph equal to the card's to 1e-9 and E312's snapped counts reproduced, a KILL comes there, not
after hours of dumping (`score_seed` takes those counts over and scores the other read-outs; with B's dump still to be made the checks come after it, only on VAL).''')
rep('''The provenance is taken before the model is loaded; a crash between the dump pickle and its sidecar leaves a dump that looks 'made by other code': move it away. Cards''',
    '''The provenance is taken before the model is loaded and written to the sidecar FIRST, the pickle goes in afterwards without overwriting (a crash leaves a sidecar without a pickle, never a pickle without provenance); a dump on a GPU waits for room as `jends_eval.py dump` does. Cards''')
rep('''The code digest (`code_sha256`: the static import closure of this file over the project's search path, 70 or so files) is printed at the start of every `all`; `--expect-code-digest D` (the `code_digest` of the rehearsal's cards, or its first 16 characters)
refuses the run before any dump when the imported lane code is not the rehearsal's, and `--expect-code-card CARD` (a card of the rehearsal) names the changed files.''',
    '''The code digest (`code_sha256`: the static import closure of this file over the project's search path, 70 or so files) is printed at the start of every `all`. THE CODE PIN: on TEST `--expect-code-card CARD` (a card of the VAL rehearsal) is REQUIRED and the expected
digest is taken from it (no second typed value); the card must be that of a full, passed rehearsal (split val, 873 frames, rehearsal check ok, no code drift: the card stores the digest of the START state, its digest is that of its own file list); `all` refuses before any dump when the imported
lane code or the environment (sha256 of data/splits.json, val_tokens.json, the `config.json` of the runs; torch / numpy / shapely versions, stored in every card) is not the card's, and names the changed files. `--expect-code-digest D` stays an optional extra check (the full digest or
its first 16 characters; an empty string refuses). On TEST the card also fixes --boot 1000, --seeds 0 1, --b-dump / --control-ends-dump / --control-card (required) and 21 scenes in the frames. At the end the repository modules this run loaded that the digest does not cover are named
(`UNDIGESTED MODULE: ...` in the log, the cards and `results.md`: not fatal, the rehearsal shows them).''')
rep('''  and `--expect-code-digest <code_digest of runs/eval/e324s_val/card_s0.json> --expect-code-card runs/eval/e324s_val/card_s0.json`, so that the code of the TEST pass is the code that was rehearsed.''',
    '''  and `--expect-code-card runs/eval/e324s_val/card_s0.json` (required; add `--expect-code-digest <code_digest of that card>` as an extra check), so that the code and the environment of the TEST pass are those that were rehearsed.''')
rep('''[--no-stack] [--expect-code-digest D [--expect-code-card CARD]]   # what run.py launches for a TEST use''',
    '''[--no-stack] [--expect-code-card CARD [--expect-code-digest D]]   # what run.py launches for a TEST use''')

open(P, "w", encoding="utf-8").write(s)
print("patched", len(s.splitlines()), "lines")
