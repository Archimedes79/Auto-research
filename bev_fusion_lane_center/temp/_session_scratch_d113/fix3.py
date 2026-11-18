import io
p = "training/data/lane_tokcache.py"
s = io.open(p, encoding="utf-8").read()
a = '''    print("lane_tokcache selftest ok")'''
b = '''    # two caches of the same frames: equal passes (NaN = NaN), a single changed value in any field fails and is named
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dirs = []
        for name_ in ("a", "b"):
            d_ = os.path.join(tmp, name_)
            os.makedirs(d_)
            json.dump(dict(rows=[1, 2, 3], tokens=["x", "y", "z"]), open(os.path.join(d_, "frames.json"), "w"))
            rng = np.random.default_rng(0)
            np.save(os.path.join(d_, "cell.npy"), np.arange(12, dtype=np.int16).reshape(3, 4))
            np.save(os.path.join(d_, "tok.npy"), rng.random((3, 4, 5)).astype(np.float16))
            sc = np.full((3, 4, 2), np.nan, np.float32)
            sc[0, 0] = [0.3, 0.4]
            np.save(os.path.join(d_, "ypair_scale.npy"), sc)
            dirs.append(d_)
        assert compare_caches(dirs[0], dirs[1]) == (True, {})
        tk_ = np.load(os.path.join(dirs[1], "tok.npy"))
        tk_[1, 2, 3] += np.float16(0.01)
        np.save(os.path.join(dirs[1], "tok.npy"), tk_)
        assert compare_caches(dirs[0], dirs[1]) == (False, {"tok": 1}) and compare_caches(dirs[0], dirs[1], n=1)[0], "only the first n frames are compared"
        sc[0, 0, 1] = np.nan
        np.save(os.path.join(dirs[1], "ypair_scale.npy"), sc)
        assert compare_caches(dirs[0], dirs[1])[1] == {"tok": 1, "ypair_scale": 1}
    print("lane_tokcache selftest ok")'''
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
