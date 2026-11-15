p = "tests/test_student_window.py"
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- 4c: the pixel noise of the older frames is not frame 0's (the model uses salt 1), tested in the model
rep('''    ok["the gradient reaches the student through the jittered newest frame"] = float(g.abs()) > 0
    return ok
''', '''    ok["the gradient reaches the student through the jittered newest frame"] = float(g.abs()) > 0
    m = build(camera_student="stub", student_trainable=True, student_frames="all").train()
    same_img = images(2, seed=9)
    b = dict(data(3), lane_img=same_img, lane_img_old=same_img[:, None].expand(2, 2, 3, 256, 704).contiguous())
    m.student.seen = []
    with torch.no_grad():
        m.camera_bevs(dict(b, img_jitter=AU.draw_jitter(2, 1.0, 5)))                   # the noise ON: identical frames differ by the noise alone
    cur_n, old_n = m.student.seen[0], m.student.seen[1].view(2, 2, 3, 256, 704)
    m.student.seen = None
    d0, d1 = float((cur_n[0] - old_n[0, 0]).abs().mean()), float((old_n[0, 0] - old_n[0, 1]).abs().mean())
    ok["model: the pixel noise of the older frames is not frame 0's (the sample's first frame has its own stream) and not each other's, and it is noise only (mean |difference| about 5.7 of 255)"] = \\
        2 < d0 < 10 and 2 < d1 < 10 and not torch.equal(cur_n[0], old_n[0, 0])
    return ok
''')

# ---- 4b: train() itself hands run_clip the step and the micro-batch
rep('''for name, v in clip_checks().items():
    check(v, name)
''', '''for name, v in clip_checks().items():
    check(v, name)


def train_jitter_checks():
    """the REAL train() (the toy of tests/support/toy_train.py: a tiny network, the data path stubbed) with img_jitter: step_fn hands run_clip the optimiser step and the micro-batch, so every micro-batch
    of every step has its own draw, and the same run draws the same again"""
    import toy_train as TT
    ok = {}

    def run(name, **kw):
        seen, saved = [], {k: getattr(T, k) for k in ("build", "forward_loss", "make_loader", "val_subset")}

        def forward(model, loss, batch, *a, **k):
            j = batch.get("img_jitter")
            seen.append(None if j is None else (j["u"].flatten().tolist(), j["noise_seed"]))
            return TT.toy_forward(model, loss, batch, *a, **k)
        T.build, T.forward_loss, T.make_loader, T.val_subset = (lambda cfg: (TT.ToyNet(), TT.ToyLoss(), True)), forward, (lambda ds, cfg, train, workers=None, generator=None: ds), (lambda ds, n: ds)
        base = dict(name=name, out_root=tmp, steps=8, accum=2, warmup=2, log_every=1, eval_every=0, save_every=4, keep=3, device="cpu", workers=0, flip_aug=0.0, seq_len=1, seed=0, lr=1e-2, grad_clip=0.5,
                    img_jitter=0.5, img_jitter_from=0.25)
        data = TT.toy_batches(20)
        try:
            with contextlib.redirect_stdout(_Capture()):
                T.train(T.TrainConfig(**dict(base, **kw)), datasets=(data, data))
        finally:
            for k, v in saved.items():
                setattr(T, k, v)
        return seen

    a, b, c = run("tj_a"), run("tj_b"), run("tj_c", seed=1)
    ok["train(): 8 steps x 2 micro-batches; no draw before step_frac 0.25 (steps 0 and 1), a draw in each of the 12 micro-batches after it"] = len(a) == 16 and all(x is None for x in a[:4]) and all(x is not None for x in a[4:])
    ok["train(): the two micro-batches of one step have different draws, and so have consecutive steps"] = all(a[i][0] != a[i + 1][0] for i in range(4, 15, 2)) and len({(tuple(x[0]), x[1]) for x in a[4:]}) == 12
    ok["train(): the same run again draws the same in every micro-batch (a function of (seed, step, micro-batch)); another seed draws other ones"] = a == b and a[4:] != c[4:] and all(x[0] != y[0] for x, y in zip(a[4:], c[4:]))
    return ok


for name, v in train_jitter_checks().items():
    check(v, name)
''')

# ---- F: markers
rep('''    ok["marker: the key-frame build writes no marker and no sweeps folder"] = LB.older_built(kroot) == {} and not os.path.exists(os.path.join(kroot, SWEEPS))''',
    '''    ok["marker: the key-frame build writes <root>/BUILT.json (key key_frames, its scenes), not a sweeps folder or an older-frame marker"] = \\
        LB.older_built(kroot) == {} and not os.path.exists(os.path.join(kroot, SWEEPS)) and set(LB.key_built(kroot)) == {LB.KEY} and LB.key_built(kroot)[LB.KEY]["scenes"] == [scene_of(toks[0])]
    ok["marker: it records the builder's constants (JPEG quality, crop): the ones of the code now"] = LB.key_built(kroot)[LB.KEY]["settings"] == LB.build_settings() and bt[key]["settings"] == LB.build_settings() \\
        and LB.build_settings()["jpeg_quality"] == LB.JPEG_QUALITY and LB.build_settings()["hw"] == [256, 704]''')
rep('''        and any("build-cache-older" in x for x in ST.older_cache_findings(kroot, cfg)) and any("names no scenes" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, train_scenes=(), val_scenes=())))
    return ok''', '''        and any("build-cache-older" in x for x in ST.older_cache_findings(kroot, cfg)) and any("names no scenes" in x for x in ST.older_cache_findings(root, dataclasses.replace(cfg, train_scenes=(), val_scenes=())))
    kcfg = dataclasses.replace(cfg, train_scenes=(scene_of(toks[0]),), val_scenes=(scene_of(toks[0]),))
    ok["startable: the key-frame marker is asked for too: one that covers the scenes is no problem, a scene it lacks, no marker and a cache made with other constants are named"] = \\
        ST.key_cache_findings(kroot, kcfg) == [] and any("lacks 1 of the config's" in x and "scene-9999" in x for x in ST.key_cache_findings(kroot, dataclasses.replace(kcfg, val_scenes=("scene-9999",)))) \\
        and any("no finished key-frame build" in x for x in ST.key_cache_findings(root, kcfg)) \\
        and any("were made with" in x for x in ST.key_cache_findings(_marker_with(kroot, settings=dict(LB.build_settings(), jpeg_quality=50)), kcfg)) \\
        and any("was made with" in x for x in ST.older_cache_findings(_marker_with(root, older=True, settings=dict(LB.build_settings(), hw=[1, 1])), cfg))
    return ok


def _marker_with(src, older=False, **extra):
    """a copy of a cache folder's BUILT.json (key frames or the older frames) with `extra` added to its entry: the folder of the copy"""
    folder = tempfile.mkdtemp(prefix="marker_copy_", dir=tmp)
    built = LB.older_built(src) if older else LB.key_built(src)
    for k in built:
        built[k] = dict(built[k], **extra)
    target = os.path.join(folder, SWEEPS) if older else folder
    os.makedirs(target, exist_ok=True)
    json.dump({"built": built}, open(os.path.join(target, LB.BUILT), "w"))
    return folder


def crash_checks():
    """a build that dies half-way leaves no marker (it is written after the last sample), keeps the files it made, and the rerun resumes and then marks"""
    ok = {}
    for older in (True, False):
        root = tempfile.mkdtemp(prefix="crash_", dir=tmp)
        d = _subset(toks[:2])
        real, calls, n_ok = d._image_uint8, [0], (2 if older else 1)                       # two older images per sample, one key frame

        def flaky(sd):
            calls[0] += 1
            if calls[0] == n_ok + 1:
                raise RuntimeError("disk full")
            return real(sd)
        d._image_uint8 = flaky
        with contextlib.redirect_stdout(_Capture()):
            died = raises(RuntimeError, lambda: LB.build_cache([d], root, 1, older=older), "disk full")
        folder = os.path.join(root, SWEEPS) if older else root
        made = {os.path.join(dp, f): os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(folder) for f in fs if f.endswith(".jpg") and (older or SWEEPS not in dp)}
        marker = (LB.older_built if older else LB.key_built)(root)
        d._image_uint8 = real
        with contextlib.redirect_stdout(_Capture()):
            LB.build_cache([d], root, 1, older=older)
        marker2 = (LB.older_built if older else LB.key_built)(root)
        ok[f"a {'older-frame' if older else 'key-frame'} build that crashes on its image after the first sample leaves no marker and keeps what it wrote; the rerun keeps those files and writes the marker"] = \\
            died and marker == {} and len(made) == n_ok and bool(marker2) and all(os.path.getmtime(k) == t_ for k, t_ in made.items())
    return ok


def lock_checks():
    """two builds marking the same cache folder at the same time keep both updates (read-modify-write under a lock). The read is made slow so that both reads happen before either write unless the lock holds"""
    import multiprocessing as mp
    ok = {}
    folder = tempfile.mkdtemp(prefix="lock_", dir=tmp)
    nusc = types.SimpleNamespace(get=lambda table, tok: {"scene_token": tok} if table == "sample" else {"name": tok})
    ds = lambda scene: types.SimpleNamespace(n_cam=3, cam_stride=2, samples=[scene], nusc=nusc)       # noqa: E731
    real_read = LB._built

    def slow_read(f):
        r = real_read(f)
        time.sleep(0.5)
        return r
    LB._built = slow_read
    try:
        procs = [mp.get_context("fork").Process(target=LB._mark_built, args=(folder, LB.KEY, [ds(sc)])) for sc in ("scene-A", "scene-B")]
        for q in procs:
            q.start()
        for q in procs:
            q.join()
    finally:
        LB._built = real_read
    ok["two builds marking one folder at once keep both scenes (lock file)"] = all(q.exitcode == 0 for q in procs) and real_read(folder)[LB.KEY]["scenes"] == ["scene-A", "scene-B"]
    return ok
''')

# ---- cli: key marker, a student-'all' config (the t26 one)
rep('''    bt = LB.older_built(root)
    bt[LB.older_key(3, 2)]["scenes"] = bt[LB.older_key(3, 2)]["scenes"][:1]
    json.dump({"built": bt}, open(os.path.join(root, SWEEPS, LB.BUILT), "w"))''', '''    cfg2_path = os.path.join(tmp, "cli_config_student.json")                              # the t26 kind: a student 'all', feature caches named (that do not exist here), n_cam 3
    json.dump({**json.load(open(cfg_path)), "feature_cache": os.path.join(tmp, "no_cam_feats"), "feature_cache2": os.path.join(tmp, "no_cam_feats_bevfusion"),
               "model": {"camera_features": "dual", "camera_student": "evit_b1", "student_frames": "all", "student_trainable": True, "cam_buffer": "recurrent"}}, open(cfg2_path, "w"))
    try:
        for cmd in ("build-cache", "build-cache-older", "check-cache", "check-cache-older"):
            with contextlib.redirect_stdout(_Capture()):
                LB._cli([cmd, cfg2_path, "--n", "2"])
        student_ok = True
    except Exception as e:                                                                # noqa: BLE001 - the finding is the message
        student_ok = f"{type(e).__name__}: {e}"
    ok[f"cli: all four commands also run on a config with a student 'all' and feature caches named (the t26 config; the builders read plain datasets) {'' if student_ok is True else student_ok}"] = student_ok is True
    ok["cli: build-cache wrote <root>/BUILT.json with both scenes of the config, check-cache compared it"] = set(LB.key_built(root)[LB.KEY]["scenes"]) == {sp["mini_train"][0], sp["mini_val"][0]} and "BUILT.json covers all 2 scenes" in say["check-cache"]
    kb = LB.key_built(root)
    kb[LB.KEY]["scenes"] = kb[LB.KEY]["scenes"][:1]
    json.dump({"built": kb}, open(os.path.join(root, LB.BUILT), "w"))
    ok["cli: check-cache fails when the key-frame marker lacks a scene of the config"] = raises(AssertionError, lambda: LB._cli(["check-cache", cfg_path, "--n", "2"]), "no finished build of key_frames")
    bt = LB.older_built(root)
    bt[LB.older_key(3, 2)]["scenes"] = bt[LB.older_key(3, 2)]["scenes"][:1]
    json.dump({"built": bt}, open(os.path.join(root, SWEEPS, LB.BUILT), "w"))''')

# ---- mini-dependent groups run
rep('''    for group in (data_checks, marker_checks, cli_checks):''', '''    for group in (data_checks, marker_checks, crash_checks, lock_checks, cli_checks):''')

# ---- H: mutants and groups
rep('''real_clip, real_collate = T.run_clip, FC.collate_cached
''', '''real_clip, real_collate = T.run_clip, FC.collate_cached
real_build = LB.build_cache


def marks_first(datasets, root, workers=1, older=False):                          # the bug: the marker written before the work
    if datasets:
        LB._mark_built(LB.older_root(root), LB.older_key(datasets[0].n_cam, datasets[0].cam_stride), datasets) if older else LB._mark_built(root, LB.KEY, datasets)
    return real_build(datasets, root, workers, older)


def clip_without_step(*a, **k):                                                    # the bug: train.step_fn does not pass step / micro
    return real_clip(*a, **{kk: vv for kk, vv in k.items() if kk not in ("step", "micro")})
''')
rep('''    "the gamma range is 1.5, not 1.25": ([(AU, "JITTER_GAMMA", 1.5)], None, JIT_G),
''', '''    "the gamma range is 1.5, not 1.25": ([(AU, "JITTER_GAMMA", 1.5)], None, JIT_G),
    "the older frames' noise uses frame 0's stream (salt 0)": ([(FC, "jitter_images", lambda img, j, salt=0: real_jitter(img, j, 0))], None, ("jmodel",)),
    "train() does not pass the step and the micro-batch to run_clip": ([(T, "run_clip", clip_without_step)], None, ("train",)),
''')
rep('''        "the marker covers every scene": ([(LB, "older_missing_scenes", lambda *a: []), (ST, "older_missing_scenes", lambda *a: [])], None, ("marker",)),
''', '''        "the marker covers every scene": ([(LB, "older_missing_scenes", lambda *a: []), (ST, "older_missing_scenes", lambda *a: [])], None, ("marker",)),
        "the key-frame marker covers every scene": ([(ST, "key_missing_scenes", lambda *a: []), (LB, "key_missing_scenes", lambda *a: [])], None, ("marker",)),
        "the marker is written before the work (a crash leaves one)": ([(LB, "build_cache", marks_first)], None, ("crash",)),
        "the marker is written without the lock": ([(sys.modules["fcntl"], "flock", lambda *a: None)], None, ("lock",)),
''')
rep('''GROUPS = {"model": model_checks, "stream": stream_batch_checks, "jfn": jitter_fn_checks, "jmodel": jitter_model_checks, "clip": clip_checks}''',
    '''GROUPS = {"model": model_checks, "stream": stream_batch_checks, "jfn": jitter_fn_checks, "jmodel": jitter_model_checks, "clip": clip_checks, "train": train_jitter_checks}''')
rep('''    GROUPS.update({"data": data_checks, "marker": marker_checks})''', '''    GROUPS.update({"data": data_checks, "marker": marker_checks, "crash": crash_checks, "lock": lock_checks})''')
rep('''import types

import numpy as np''', '''import fcntl  # noqa: F401 - the lock of the marker (section H plants a bug in it)
import types

import numpy as np''')
open(p, "w").write(s)
print("ok")
