"""E358 scratch: mutation evidence for the NEW lines of jwire_data.py / jwire_train.py / jwire_verdict.py / jwire_check.py.

    wsl bash temp/e358/run_scratch.sh temp/_session_scratch_e358/mutate.py <jwire_data|jwire_train|jwire_verdict|jwire_check>

Each mutant is the module source with ONE replacement, executed as a fresh module (the files in the tree are never touched); the tests of the module (`_selftest`, for jwire_train `_test_pair_run` + `_test_ends`)
run against it: KILLED = the tests raised, SURVIVED = they passed. A replacement that does not apply exactly once is reported as NOT APPLIED.
"""
import contextlib
import io
import os
import sys
import time
import types

ROOT = os.environ["BFLC_ROOT"]
FILES = dict(jwire_data="evaluation/jwire_data.py", jwire_train="training/train/jwire_train.py", jwire_verdict="evaluation/jwire_verdict.py", jwire_check="evaluation/jwire_check.py")

M = dict(
    jwire_data=[
        ("check_ends: no membership test", """    assert ends in ENDS, f"ends {ends!r}: one of {ENDS}"\n""", ""),
        ("check_ends: att accepted on E312's pair run", """or resolve_pair_run(seed, pair_run) != default_pair_run(seed), f"ends att needs""", """or True, f"ends att needs"""),
        ("record_of: att ignored", """    attach = check_ends(ends) == "att"\n""", """    attach = check_ends(ends) == "never"\n"""),
        ("record_of: no att_pos assertion", """    assert not attach or ("att_pos" in e and "att_dir" in e), "ends att: the frame""", """    assert True or ("att_pos" in e and "att_dir" in e), "ends att: the frame"""),
        ("record_of: attach inverted", "pairs = QEV.make_pairs(e, ia, ib, k, attach=attach)", "pairs = QEV.make_pairs(e, ia, ib, k, attach=not attach)"),
        ("record_of: GT match on the attached ends", """    m_in = match(e["kp_pos"][:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])""", """    m_in = match((e["att_pos"] if attach else e["kp_pos"])[:k], e["kp_dir"][:k], valid[0], g["ent"], g["ent_m"])"""),
        ("val_records: ends not passed to record_of", "gt[f.row], thr, f.row, f.scene, ends))", "gt[f.row], thr, f.row, f.scene))"),
        ("val_records: no check_ends", """    check_ends(ends, seed, run)\n    check_seed""", """    check_seed"""),
        ("live_records: ends not passed to record_of", """int(b["row"][i]), -1, ends))""", """int(b["row"][i]), -1))"""),
        ("live_records: no check_ends", """    check_ends(ends, seed, pair_run)\n    import torch""", """    import torch"""),
        ("live_records: frame_dict without ends", "e = frame_dict(o, i, ends)", "e = frame_dict(o, i)"),
        ("frame_dict: attached ends of item 0", """att_pos=o["att_pos"][i].float()""", """att_pos=o["att_pos"][0].float()"""),
        ("frame_dict: attached dir of item 0", """att_dir=o["att_dir"][i].float()""", """att_dir=o["att_dir"][0].float()"""),
        ("frame_dict: no assertion", """    assert "att_pos" in o and "att_dir" in o, "ends att: the output""", """    assert True or "att_pos" in o and "att_dir" in o, "ends att: the output"""),
        ("frame_dict: silent fall back to the keypoints", """e.update(att_pos=o["att_pos"][i].float().cpu().numpy(), att_dir=o["att_dir"][i].float().cpu().numpy())""", """e.update(att_pos=e["kp_pos"], att_dir=e["kp_dir"])"""),
        ("frame_dict: att ends also for kp", """    if check_ends(ends) == "att":\n        assert "att_pos" """, """    if check_ends(ends) in ENDS:\n        assert "att_pos" """),
        ("cache_path: no att suffix", """{'_ends_' + ends if ends != 'kp' else ''}""", """{''}"""),
        ("cache_path: suffix also for kp", """{'_ends_' + ends if ends != 'kp' else ''}""", """{'_ends_' + ends}"""),
        ("cache_path: no check_ends", """    check_ends(ends, seed, pair_run)\n    other = """, """    other = """),
        ("meta_of: ends always in the meta", """    if check_ends(ends, seed, pair_run) != "kp":\n        meta["ends"] = ends""", """    check_ends(ends, seed, pair_run)\n    meta["ends"] = ends"""),
        ("meta_of: ends never in the meta", """        meta["ends"] = ends\n""", """        pass\n"""),
        ("save_records: meta without ends", "meta=meta_of(seed, pair_run, ends), records=records", "meta=meta_of(seed, pair_run), records=records"),
        ("load_records: want without ends", "    want = meta_of(seed, pair_run, ends)\n", "    want = meta_of(seed, pair_run)\n"),
        ("cached_records: load without ends", "        return load_records(path, seed, pair_run, ends)\n", "        return load_records(path, seed, pair_run)\n"),
        ("cached_records: save without ends", "    save_records(path, recs, seed, pair_run, ends)\n", "    save_records(path, recs, seed, pair_run)\n"),
        ("pilot_stats: NEAR_M 0.45", "NEAR_M = 0.5", "NEAR_M = 0.45"),
        ("pilot_stats: NEAR_M 0.55", "NEAR_M = 0.5", "NEAR_M = 0.55"),
        ("pilot_stats: near05 counts every end", """for i in inst if i["kind"] == "oracle"]""", """for i in inst]"""),
        ("pilot_stats: near05 of the farthest place", """float(i["feat"][0, 0]) * JH.REACH <= NEAR_M""", """float(i["feat"][-1, 0]) * JH.REACH <= NEAR_M"""),
        ("pilot_stats: near05 key missing", """near05=q(near, np.mean), counts=tot)""", """counts=tot)"""),
        ("main val: ends not passed to val_records", "val_records(a.seed, resolve_pair_run(a.seed, pr), a.thr, a.max, a.ends), a.ends)", "val_records(a.seed, resolve_pair_run(a.seed, pr), a.thr, a.max), a.ends)"),
        ("main val: ends not passed to cache_path", """path = cache_path("val", a.seed, a.max, a.thr, pr, a.ends)""", """path = cache_path("val", a.seed, a.max, a.thr, pr)"""),
        ("main val: ends not passed to cached_records", "a.thr, a.max, a.ends), a.ends)", "a.thr, a.max, a.ends))"),
        ("main live: no check_ends", "        check_ends(a.ends, a.seed, pr)", "        pass"),
        ("main live: ends not passed to live_records", "pair_run=pr, ends=a.ends), a.ends)", "pair_run=pr), a.ends)"),
        ("main live: ends not passed to cache_path", "cache_path(a.split, a.seed, a.max, 0.1, pr, a.ends)", "cache_path(a.split, a.seed, a.max, 0.1, pr)"),
        ("main live: ends not passed to cached_records", "pair_run=pr, ends=a.ends), a.ends)", "pair_run=pr, ends=a.ends))"),
        ("main val: --ends without choices", """    s.add_argument("--ends", default="kp", choices=ENDS, """, """    s.add_argument("--ends", default="kp", """),
        ("main live: --ends without choices", """    t.add_argument("--ends", default="kp", choices=ENDS, """, """    t.add_argument("--ends", default="kp", """),
        ("main val: --ends default att", """    s.add_argument("--ends", default="kp", """, """    s.add_argument("--ends", default="att", """),
    ],
    jwire_train=[
        ("run_target: no check_ends", "    JD.check_ends(ends, seed, pair_run)\n", ""),
        ("run_pipeline: run_target without ends", "run_target(seed, name, pair_run, ends)", "run_target(seed, name, pair_run)"),
        ("run_pipeline: val cache path without ends", """vpath = JD.cache_path("val", seed, pair_run=pair_run, ends=ends)""", """vpath = JD.cache_path("val", seed, pair_run=pair_run)"""),
        ("run_pipeline: train cache path without ends", """tpath = JD.cache_path("train", seed, max_train, pair_run=pair_run, ends=ends)""", """tpath = JD.cache_path("train", seed, max_train, pair_run=pair_run)"""),
        ("run_pipeline: val_records without ends", "lambda: JD.val_records(seed, pair_run, ends=ends), ends)", "lambda: JD.val_records(seed, pair_run), ends)"),
        ("run_pipeline: val cached_records without ends", "lambda: JD.val_records(seed, pair_run, ends=ends), ends)", "lambda: JD.val_records(seed, pair_run, ends=ends))"),
        ("run_pipeline: live_records without ends", "workers, pair_run=pair_run, ends=ends), ends)", "workers, pair_run=pair_run), ends)"),
        ("run_pipeline: train cached_records without ends", "workers, pair_run=pair_run, ends=ends), ends)", "workers, pair_run=pair_run, ends=ends))"),
        ("run_pipeline: no ends in the json", "info = dict(seed=seed, pair_run=pair_run, ends=ends, train_frames", "info = dict(seed=seed, pair_run=pair_run, train_frames"),
        ("run_pipeline: no ends in the head checkpoints", "args=dict(seed=seed, pair_run=pair_run, ends=ends, head_seed=h,", "args=dict(seed=seed, pair_run=pair_run, head_seed=h,"),
        ("run_pipeline: no ends in the wn checkpoint", """args=dict(seed=seed, pair_run=pair_run, ends=ends, head_seed="wn",""", """args=dict(seed=seed, pair_run=pair_run, head_seed="wn","""),
        ("run_pipeline: wn trained with the jitter", "nf, nm, nl, no, nn, _ = tensors(train)", "nf, nm, nl, no, nn, _ = tensors(train, JD.make_jitter(scale, 0))"),
        ("run_pipeline: wn head seed 7", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), epochs, seed=7)"),
        ("run_pipeline: wn rows not repeated", "fit(nf.repeat(draws, 1, 1), nm.repeat(draws, 1), nl.repeat(draws), epochs, seed=0)", "fit(nf, nm, nl, epochs, seed=0)"),
        ("run_pipeline: wn written over head 0", """run.path("model_wn.pt")""", """run.path("model.pt")"""),
        ("run_pipeline: wn judged on the jittered ends", "at = agree(predict(w, nf, nm), nl, no, nn)", "at = agree(predict(w, feat, mask), label, orc, npl)"),
        ("run_pipeline: wn only the distance columns", "nl.repeat(draws), epochs, seed=0)", "nl.repeat(draws), epochs, seed=0, cols=DIST_ONLY)"),
        ("run_pipeline: wn not in the json", """    out["no_jitter_head"] = dict(val=av, train=at)\n""", ""),
        ("main run: ends not passed", "pair_run=a.pair_run, ends=a.ends)", "pair_run=a.pair_run)"),
        ("main cv: cache path without ends", """JD.cache_path("val", a.seed, pair_run=a.pair_run, ends=a.ends)""", """JD.cache_path("val", a.seed, pair_run=a.pair_run)"""),
        ("main cv: load without ends", "a.seed, a.pair_run, a.ends)\n    res = cv", "a.seed, a.pair_run)\n    res = cv"),
        ("main run: --ends without choices", """    r.add_argument("--ends", default="kp", choices=JD.ENDS, """, """    r.add_argument("--ends", default="kp", """),
        ("main run: --ends default att", """    r.add_argument("--ends", default="kp", """, """    r.add_argument("--ends", default="att", """),
    ],
    jwire_verdict=[
        ("stem: primary reads the snapped names", """return "snap_nogeom_attach_" if read_out == "primary" else "snap_\"""", """return "snap_\""""),
        ("check_inputs: attach as before", """("attach", bool(c.get("attach")) == primary)""", """("attach", not c.get("attach"))"""),
        ("check_inputs: attach not checked", """("attach", bool(c.get("attach")) == primary)""", """("attach", True)"""),
        ("check_inputs: snap_nogeom not checked", """("snap_nogeom", got["snap_nogeom"] == primary)""", """("snap_nogeom", True)"""),
        ("check_inputs: snap_nogeom always true", """snap_nogeom="snap_nogeom" in (c.get("snap_options") or []))""", """snap_nogeom=True)"""),
        ("check_inputs: head kind not checked", """    if wa.get("ends", "kp") != READ_OUT_ENDS[a.read_out]:""", """    if False:"""),
        ("check_inputs: no field read as att", """wa.get("ends", "kp") != READ_OUT_ENDS[a.read_out]:""", """wa.get("ends", "att") != READ_OUT_ENDS[a.read_out]:"""),
        ("check_inputs: bootstrap names of the snapped cards", """(run, f"ends_card_{a.tag}_{stem(a.read_out)}thr0.1"), (run, f"ends_card_{stem(a.read_out)}thr0.1")""", """(run, f"ends_card_{a.tag}_snap_thr0.1"), (run, "ends_card_snap_thr0.1")"""),
        ("check_inputs: primary = False", """    primary = a.read_out == "primary"\n""", """    primary = False\n"""),
        ("READ_OUT_ENDS swapped", """READ_OUT_ENDS = dict(snap="kp", primary="att")""", """READ_OUT_ENDS = dict(snap="att", primary="kp")"""),
        ("judge: share difference one-sided", "abs(near[0] - near[1]) > SHARE_LIMIT", "(near[0] - near[1]) > SHARE_LIMIT"),
        ("judge: SHARE_LIMIT 0.20", "SHARE_LIMIT, POINT_TOL = 0.02, 0.7, 0.10, 1e-4", "SHARE_LIMIT, POINT_TOL = 0.02, 0.7, 0.20, 1e-4"),
        ("judge: SHARE_LIMIT 0.05", "SHARE_LIMIT, POINT_TOL = 0.02, 0.7, 0.10, 1e-4", "SHARE_LIMIT, POINT_TOL = 0.02, 0.7, 0.05, 1e-4"),
        ("judge: share check on a missing number", "        if None not in near and abs", "        if True and abs"),
        ("judge: no share check", "        if None not in near and abs(near[0] - near[1]) > SHARE_LIMIT:", "        if False:"),
        ("read_seed: cards of the snapped names", """{tag + '_' if tag else ''}{stem(a.read_out)}thr0.1.json""", """{tag + '_' if tag else ''}snap_thr0.1.json"""),
        ("render: head label", """{'primary' if a.read_out == 'primary' else 'snapped'}: " +""", """snapped: " +"""),
        ("render: no wn line", """        nj = wa.get("no_jitter_head")""", """        nj = None"""),
        ("render: read-out not named", ", tag {a.tag}, read-out {a.read_out}\"]", ", tag {a.tag}\"]"),
        ("main: --read-out default primary", """ap.add_argument("--read-out", default="snap",""", """ap.add_argument("--read-out", default="primary","""),
    ],
    jwire_check=[
        ("read_out_ends: ends alone ignored", """read_out = read_out or {v: k for k, v in READ_OUT_ENDS.items()}.get(ends, "snap")""", """read_out = read_out or "snap\""""),
        ("read_out_ends: no agreement check", """    assert ends in (None, READ_OUT_ENDS[read_out]), f"--read-out""", """    assert True or ends in (None, READ_OUT_ENDS[read_out]), f"--read-out"""),
        ("READ_OUT_ENDS swapped", """READ_OUT_ENDS = dict(snap="kp", primary="att")""", """READ_OUT_ENDS = dict(snap="att", primary="kp")"""),
        ("twin_check: names without nogeom", "attach=primary, snap_nogeom=primary))", "attach=primary))"),
        ("twin_check: names without attach", "attach=primary, snap_nogeom=primary))", "snap_nogeom=primary))"),
        ("twin_check: no options check", """        if bool(c.get("attach")) != primary or ("snap_nogeom" in (c.get("snap_options") or [])) != primary:""", """        if False:"""),
        ("twin_check: no nogeom option check", """ or ("snap_nogeom" in (c.get("snap_options") or [])) != primary:""", """:"""),
        ("twin_check: no attach option check", """        if bool(c.get("attach")) != primary or ("snap_nogeom" """, """        if ("snap_nogeom" """),
        ("live_check: cache path without ends", """JD.cache_path("val", seed, pair_run=pair_run, ends=ends)""", """JD.cache_path("val", seed, pair_run=pair_run)"""),
        ("live_check: load without ends", "JD.load_records(path, seed, pair_run, ends)", "JD.load_records(path, seed, pair_run)"),
        ("live_check: live pass without ends", "device=dev, pair_run=pair_run, ends=ends)", "device=dev, pair_run=pair_run)"),
        ("live_check: message without --ends", " --pair-run {pair_run} --ends {ends}` builds", " --pair-run {pair_run}` builds"),
        ("main: live without ends", "live_check(a.seed, a.pair_run, a.max, a.workers, a.device, ends)", "live_check(a.seed, a.pair_run, a.max, a.workers, a.device)"),
        ("main: twin without read_out", "twin_check(a.run, a.seed, a.tag, a.thr, read_out)", "twin_check(a.run, a.seed, a.tag, a.thr)"),
        ("main: --ends without choices", """choices=tuple(READ_OUT_ENDS.values()), help""", """help"""),
    ],
)


def load(name, src):
    mod = types.ModuleType(name)
    mod.__file__ = os.path.join(ROOT, FILES[name])
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def run_tests(name, mod):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if name == "jwire_train":
            mod._test_pair_run()
            mod._test_ends()
        else:
            mod._selftest()


def main(name):
    import jwire_data  # noqa: F401  (the real modules first: the mutant replaces only its own entry in sys.modules)
    import jwire_train  # noqa: F401
    import jwire_check  # noqa: F401
    import jwire_verdict  # noqa: F401
    real = sys.modules[name]
    src = open(os.path.join(ROOT, FILES[name]), encoding="utf-8").read()
    t0 = time.time()
    run_tests(name, load(name, src))
    sys.modules[name] = real
    print(f"{name}: the unmutated tests pass ({time.time() - t0:.0f} s)", flush=True)
    killed, survived, skipped = [], [], []
    for label, old, new in M[name]:
        n = src.count(old)
        if n < 1:
            skipped.append(label)
            print(f"  NOT APPLIED ({n}x) {label}", flush=True)
            continue
        t0 = time.time()
        try:
            mod = load(name, src.replace(old, new, 1))
            run_tests(name, mod)
            res, why = "SURVIVED", ""
        except BaseException as e:  # noqa: BLE001  (any failure of the tests kills the mutant)
            res, why = "KILLED", f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}"
        finally:
            sys.modules[name] = real
        (killed if res == "KILLED" else survived).append(label)
        print(f"  {res:8s} ({time.time() - t0:3.0f} s) {label}  {('<- ' + why) if why else ''}", flush=True)
    print(f"{name}: {len(killed)} killed, {len(survived)} survived, {len(skipped)} not applied, of {len(M[name])}")
    for s in survived:
        print("  SURVIVOR:", s)


if __name__ == "__main__":
    main(sys.argv[1])
