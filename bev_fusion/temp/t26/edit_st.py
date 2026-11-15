p = "training/startable.py"
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("from lane_branch import older_built, older_key, older_missing_scenes, older_root      # noqa: E402",
    "from lane_branch import KEY, build_settings, key_built, key_missing_scenes, older_built, older_key, older_missing_scenes, older_root      # noqa: E402")
rep("SMOKE = dict(steps=20, warmup=5, eval_every=10, save_every=10, keep=1, log_every=5)",
    "SMOKE = dict(steps=20, warmup=5, eval_every=10, save_every=10, keep=1, log_every=5, keep_steps=[])      # keep_steps: the engine refuses a step above `steps` or off the save_every grid (t26 lists 6000 .. 36000)")
rep("""        si =_abs(mc.student_init, root)""", """        si = _abs(mc.student_init, root)""")
rep("""            problems += older_cache_findings(_abs(cfg.lane_images, root), cfg)""", """            problems += older_cache_findings(_abs(cfg.lane_images, root), cfg) + key_cache_findings(_abs(cfg.lane_images, root), cfg)""")
rep('''    if key not in built:
        return [f"model.student_frames 'all': {cfg.lane_images}/sweeps has no finished build for {key} (BUILT.json lists {sorted(built) or 'nothing'}; python training/lane_branch.py build-cache-older <config>)"]
    missing = older_missing_scenes(''', '''    if key not in built:
        return [f"model.student_frames 'all': {cfg.lane_images}/sweeps has no finished build for {key} (BUILT.json lists {sorted(built) or 'nothing'}; python training/lane_branch.py build-cache-older <config>)"]
    if built[key].get("settings", build_settings()) != build_settings():
        return [f"model.student_frames 'all': the build of {key} in {cfg.lane_images}/sweeps was made with {built[key]['settings']}, the builder now makes {build_settings()}: build it again in a new folder"]
    missing = older_missing_scenes(''')
rep('''def check(path, root=ROOT, require_clean=False, expect_gt=""):''', '''def key_cache_findings(lane_images, cfg):
    """problems of the key-frame image cache of a student_frames 'all' config: <lane_images>/BUILT.json must hold a FINISHED key-frame build that covers every scene of the config (the older-frame
    cache has its own marker). Other configs with lane_images are not asked: their caches were built before the marker existed"""
    scenes, built = set(cfg.train_scenes) | set(cfg.val_scenes), key_built(lane_images)
    if not scenes:
        return []                                                       # older_cache_findings says so
    if KEY not in built:
        return [f"model.student_frames 'all': {cfg.lane_images} has no finished key-frame build (BUILT.json lists {sorted(built) or 'nothing'}; python training/lane_branch.py build-cache <config>: "
                "resumable, it writes the marker when every sample is through)"]
    if built[KEY].get("settings", build_settings()) != build_settings():
        return [f"model.student_frames 'all': the key frames in {cfg.lane_images} were made with {built[KEY]['settings']}, the builder now makes {build_settings()}"]
    missing = key_missing_scenes(lane_images, scenes)
    return [f"model.student_frames 'all': the finished key-frame build in {cfg.lane_images} lacks {len(missing)} of the config's {len(scenes)} scenes, e.g. {missing[:3]} "
            "(python training/lane_branch.py build-cache <config> with this config)"] if missing else []


def check(path, root=ROOT, require_clean=False, expect_gt=""):''')
rep('''    d.update(SMOKE, name="smoke_" + base)
''', '''    d.update(SMOKE, name="smoke_" + base)
    d["lr_groups"] = {k: dict(v, warmup=min(v["warmup"], SMOKE["warmup"])) if "warmup" in v else v for k, v in d.get("lr_groups", {}).items()}      # a group's own warm-up (t26: 3000) is longer than the run
''')
rep('''    """Write the smoke twin of a config: the same model and loss except the loss warm-up (SMOKE_LOSS), 20 steps; returns its path."""''',
    '''    """Write the smoke twin of a config: the same model and loss except the loss warm-up (SMOKE_LOSS), 20 steps, no keep_steps, the warm-up of an lr group cut to the run's; returns its path."""''')
# selftest: the smoke twin of a t26-like config
rep('''        assert "warm-up is replaced" in d["_about"] and T.read_config(s).steps == 20''', '''        assert "warm-up is replaced" in d["_about"] and T.read_config(s).steps == 20
        s2 = smoke_config(write(name="long", steps=36000, save_every=500, keep_steps=[6000, 12000, 36000], lr_groups={"student": {"lr": 3e-5, "warmup": 3000}, "motion_block": {"lr": 1e-4}}),
                          os.path.join(tmp, "smk"))
        c2 = T.read_config(s2)                                                                       # the engine refuses a keep_steps entry above `steps` / off the save_every grid at its start
        assert c2.steps == 20 and not c2.keep_steps and all(0 < k <= c2.steps and k % c2.save_every == 0 for k in c2.keep_steps), c2.keep_steps
        assert c2.lr_groups == {"student": {"lr": 3e-5, "warmup": 5}, "motion_block": {"lr": 1e-4}}, c2.lr_groups''')
open(p, "w").write(s)
