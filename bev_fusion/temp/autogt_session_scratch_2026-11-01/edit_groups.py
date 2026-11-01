p = "autogt/suite.py"
s = open(p, encoding="utf-8").read()
s = s.replace("    light=True: a memory-light devkit (CAM_FRONT + radars only, autogt/light_nusc.py): v1.0-trainval in ~4 GB instead of 8+ GB\n",
              "    light=True: a memory-light devkit (CAM_FRONT + radars only, autogt/light_nusc.py); scenes=[names]: only these sequences are\n"
              "    loaded (v1.0-trainval: ~0.7 GB for 85 sequences instead of 8+ GB for all)\n")
s = s.replace("def __init__(self, dataroot, version, grid=None, ss=4, light=False, **options):", "def __init__(self, dataroot, version, grid=None, ss=4, light=False, scenes=None, **options):")
s = s.replace('options["nusc"] = light_nuscenes(dataroot, version)', 'options["nusc"] = light_nuscenes(dataroot, version, scenes)')
open(p, "w", encoding="utf-8").write(s)

p = "autogt/report.py"
s = open(p, encoding="utf-8").read()
s = s.replace("import datetime\nimport multiprocessing as mp", "import datetime\nimport gc\nimport multiprocessing as mp")
s = s.replace("from .suite import AutoGT  # noqa: E402", "from .light_nusc import scene_names  # noqa: E402\nfrom .suite import AutoGT  # noqa: E402")
s = s.replace("- `--light` (automatic for trainval): the memory-light devkit (autogt/light_nusc.py), about 4 GB instead of 8+ GB",
              "- `--light` (automatic for trainval): the memory-light devkit (autogt/light_nusc.py); the sequences are processed in groups of\n"
              "  `GROUP` (60), each group loads only its own records: about 1 GB instead of 8+ GB for the full devkit")
s = s.replace("PAGE_LINES = 78", "PAGE_LINES = 78\nGROUP = 60")
a = s.index("    gen = AutoGT(dataroot, version, light=light, **options)")
b = s.index("    ck = Checks(gen.grid)\n    failed")
new = '''    names = scenes or (scene_names(dataroot, version) if light else AutoGT(dataroot, version, **options).scenes())
    gen0 = AutoGT(dataroot, version, light=light, scenes=names[:1] if light else None, **options)
    sig = gen0.signature()
    grid = gen0.grid
    sig_dir = os.path.join(data_dir, sig) if data_dir else None
    if sig_dir:
        os.makedirs(sig_dir, exist_ok=True)
    pick = set(names[i] for i in np.unique(np.linspace(0, len(names) - 1, min(samples, len(names))).round().astype(int))) if samples else set()
    results, t0, done = {}, time.time(), 0
    groups = [names[i:i + GROUP] for i in range(0, len(names), GROUP)] if light else [names]
    for grp in groups:
        gen = AutoGT(dataroot, version, light=light, scenes=grp if light else None, **options) if (light or gen0 is None) else gen0
        _STATE["gen"], _STATE["cfg"] = gen, {"sig_dir": sig_dir, "stride": stride, "pick": pick}
        its = None
        if workers > 1:
            pool = mp.get_context("fork").Pool(workers)
            its = pool.imap_unordered(_scene_task, grp)
        else:
            its = (_scene_task(n) for n in grp)
        for r in its:
            results[r[0]] = r
            done += 1
            if verbose:
                print(f"[{done}/{len(names)}] {r[0]} {'FAILED ' + r[3] if r[3] else 'ok'}  {time.time() - t0:.0f} s, "
                      f"ETA {(time.time() - t0) / done * (len(names) - done) / 60:.0f} min", flush=True)
        if workers > 1:
            pool.close()
            pool.join()
        _STATE.clear()
        del gen, its
        gc.collect()
    # the sample pictures and the radar field of view need the devkit of a few sequences only
    keep = sorted(n for n in names if results[n][2] is not None) or names[:1]
    gen = AutoGT(dataroot, version, light=light, scenes=keep if light else None, **options)
'''
s = s[:a] + new + s[b:]
s = s.replace("    ck = Checks(gen.grid)\n    failed", "    ck = Checks(grid)\n    failed")
s = s.replace('''    fov = radar_fov(gen, [n for n in names if n not in failed], stride)
    ho = leave_one_radar_out(gen, [n for n in names if n not in failed], stride) if holdout else None''', '''    fov = radar_fov(gen, [n for n in keep if n not in failed], stride)
    ho = leave_one_radar_out(gen, [n for n in names if n not in failed], stride) if (holdout and not light) else None''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
