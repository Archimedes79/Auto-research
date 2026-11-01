p = "autogt/cli.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    r.add_argument("--dataroot", default="~/data/nuscenes")
    r.add_argument("--version", default="v1.0-mini")''', '''    r.add_argument("--dataroot", default=None, help="default ~/data/nuscenes (mini) or ~/data/nuscenes_full (trainval)")
    r.add_argument("--version", default="v1.0-mini")
    r.add_argument("--workers", type=int, default=1, help="processes (fork, one sequence per task); a finished sequence is saved, reruns resume")
    r.add_argument("--light", action="store_true", help="memory-light devkit (automatic for trainval)")''')
s = s.replace('''        report(os.path.expanduser(a.dataroot), a.version, out, data, a.scenes, a.stride, a.holdout, a.samples)''', '''        root = a.dataroot or ("~/data/nuscenes_full" if "trainval" in a.version else "~/data/nuscenes")
        report(os.path.expanduser(root), a.version, out, data, a.scenes, a.stride, a.holdout, a.samples, a.workers,
               a.light or "trainval" in a.version)''')
s = s.replace('''  python -m autogt report [--version v1.0-mini] [--out REPORT.pdf] [--data DIR | --no-data] [--scenes ..] [--stride N] [--samples N] [--holdout]''',
              '''  python -m autogt report [--version v1.0-mini|v1.0-trainval] [--out REPORT.pdf] [--data DIR | --no-data] [--scenes ..] [--stride N]
                          [--samples N] [--holdout] [--workers N] [--light]''')
open(p, "w", encoding="utf-8").write(s)

p = "autogt/selftest.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''    from . import clean as cleanmod''', '''    # 7b the memory-light devkit gives the identical GT, and a parallel run of the report equals a serial one and resumes
    from .suite import AutoGT
    full = AutoGT(root, "v1.0-mini", ss=4)
    light = AutoGT(root, "v1.0-mini", ss=4, light=True)
    tk = full.keyframes("scene-0061")[8]
    a, b2 = full.keyframe(tk), light.keyframe(tk)
    assert all(np.array_equal(a[k_], b2[k_]) for k_ in a if k_ != "meta"), "light devkit must give identical GT"
    assert len(light.nusc.sample_data) < 0.7 * len(full.nusc.sample_data)
    with tempfile.TemporaryDirectory() as d:
        kw = dict(scenes=["scene-0061", "scene-0103", "scene-1077"], stride=20, samples=3, verbose=False)
        r1 = report(root, "v1.0-mini", os.path.join(d, "a.pdf"), os.path.join(d, "d1"), workers=1, **kw)
        r2 = report(root, "v1.0-mini", os.path.join(d, "b.pdf"), os.path.join(d, "d2"), workers=2, light=True, **kw)
        assert r1["keyframes"] == r2["keyframes"] and r1["samples"] == r2["samples"] and not r2["failed"], (r1, r2)
        sd2 = os.path.join(d, "d2", r2["signature"])
        assert len([f for f in os.listdir(sd2) if f.startswith("_scene_")]) == 3
        r3 = report(root, "v1.0-mini", os.path.join(d, "c.pdf"), os.path.join(d, "d2"), workers=2, light=True, **kw)       # resumes
        assert r3["keyframes"] == r1["keyframes"] and r3["written"] == 0, r3
    from . import clean as cleanmod''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
