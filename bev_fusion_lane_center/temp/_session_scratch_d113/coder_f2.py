p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''    toy_cache_test()
    frame_counts_test()
    print("lane_past_data selftest ok")''', '''    ext_ = dict(rel_R=np.stack([np.eye(2) * (i + 1) for i in range(4)]), rel_t=np.arange(8.0).reshape(4, 2))                  # distinct entries: frame i has ITS entry (not its previous frame's)
    R_, t_ = pose_of(ext_, 2)
    assert np.array_equal(R_, np.eye(2) * 3) and np.array_equal(t_, [4.0, 5.0]), "pose_of reads the entry of the frame itself"
    gd, bd = dict(n=30, median=0.5, p90=1.5, p99=3.0), dict(n=30, median=1.0, p90=2.5, p99=5.0)
    lt = dict(loader=gd, identity=bd, entry_shifted=bd, frame_shifted=bd)
    assert all(loader_fixture_checks(lt).values()) and len(loader_fixture_checks(lt)) == 4
    for name_, t2 in dict(loader=dict(lt, loader=dict(gd, p90=1.6)), identity=dict(lt, identity=gd), entry=dict(lt, entry_shifted=gd), frame=dict(lt, frame_shifted=gd), empty=dict(lt, loader=dict(gd, n=0))).items():
        assert not all(loader_fixture_checks(t2).values()), f"the loader fixture accepted bad {name_}"
    names = []
    _atomic(os.path.join(tempfile.gettempdir(), "past.json"), lambda tmp: (names.append(os.path.basename(tmp)), open(tmp, "w").close()))
    _atomic(os.path.join(tempfile.gettempdir(), "prev.npy"), lambda tmp: (names.append(os.path.basename(tmp)), np.save(tmp, np.zeros(1))))
    assert names == [f"past.{os.getpid()}.tmp.json", f"prev.{os.getpid()}.tmp.npy"], names
    os.remove(os.path.join(tempfile.gettempdir(), "past.json"))
    os.remove(os.path.join(tempfile.gettempdir(), "prev.npy"))
    toy_cache_test()
    frame_counts_test()
    print("lane_past_data selftest ok")''')
rep('''        info = build(d, pf, check=False)
        assert info["with_past"] == 7''', '''        dv = os.path.join(tmp, "bX_0_K8_sig_v1", "val")                                                   # the TEST guard of `build` (check=True): a VAL cache that is not data/splits.json['val'] is refused before anything is written
        shutil.copytree(d, dv)
        json.dump(dict(split="val"), open(os.path.join(dv, "meta.json"), "w"))
        refuses(lambda: build(dv, pf, check=True), "splits.json")
        assert not os.path.exists(os.path.join(dv, "past.json")) and not os.path.exists(os.path.join(dv, "prev.npy")), "a refused build wrote files"
        info = build(d, pf, check=False)
        assert info["with_past"] == 7''')
rep("import os\nimport sys\nimport tempfile", "import os\nimport shutil\nimport sys\nimport tempfile")
open(p, "w", encoding="utf-8").write(s)
print("ok")
