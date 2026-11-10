def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ("from .results import Locked, ResultSet, result_id, writer_lock\n", "from .results import Locked, ResultSet, decode, encode, result_id, writer_lock\n"),
    ('''    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.pkl"), "wb") as f:
        f.write(b"not a frame")''', '''    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.npz"), "wb") as f:
        f.write(b"not a frame")'''),
    ('''    assert ResultSet.create(os.path.join(tmp, rid), ident).has(refs[2]), "the stored set of the same identity is found again"''', '''    assert ResultSet.create(os.path.join(tmp, rid), ident).has(refs[2]), "the stored set of the same identity is found again"
    # the frame file is a plain npz that any numpy reads (no pickle): the arrays and tree.json
    with np.load(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), allow_pickle=False) as z:
        assert "tree.json" in z.files and "a0" in z.files and np.array_equal(z["a0"], np.arange(6, dtype=np.float32).reshape(2, 3))
        assert "$d" in json.loads(bytes(z["tree.json"]).decode("utf-8"))'''),
    ('''def check_results():''', '''def check_codec():
    """encode / decode: everything a prediction holds comes back with its types; nothing that needs pickle is stored"""
    obj = {"dt": {-0.5: {"prob": {"a": np.arange(6, dtype=np.float32).reshape(2, 3)}, "dets": [{"x": np.float32(1.5), "n": 3, "c": "car", "ok": True, "box": (1.0, 2.0)}], "none": None},
                  0.0: {"vhat": np.zeros((2, 2), np.float16)}, 1: {"i": np.int64(7), "m": np.array([True, False])}},
           "nan": float("nan"), "inf": float("-inf"), "scalar": np.array(2.5), "empty": np.zeros((0, 3), np.float32), "ints": [1, 2, [3]]}
    tree, arrays = encode(obj)
    json.dumps(tree, allow_nan=False)                                          # strict json: no bare NaN
    back = decode(json.loads(json.dumps(tree)), arrays)
    assert list(back["dt"]) == [-0.5, 0.0, 1] and [type(k) for k in back["dt"]] == [float, float, int], "keys keep their type and order"
    p = back["dt"][-0.5]
    assert np.array_equal(p["prob"]["a"], obj["dt"][-0.5]["prob"]["a"]) and p["prob"]["a"].dtype == np.float32 and p["none"] is None
    d = p["dets"][0]
    assert d == {"x": np.float32(1.5), "n": 3, "c": "car", "ok": True, "box": (1.0, 2.0)} and type(d["x"]) is np.float32 and type(d["box"]) is tuple and d["ok"] is True
    assert back["dt"][0.0]["vhat"].dtype == np.float16 and back["dt"][1]["m"].dtype == bool and type(back["dt"][1]["i"]) is np.int64
    assert back["nan"] != back["nan"] and back["inf"] == float("-inf") and back["scalar"].shape == () and back["empty"].shape == (0, 3) and back["ints"] == [1, 2, [3]]
    _raises(TypeError, "object array", encode, {"a": np.array([None, 1], dtype=object)})
    _raises(TypeError, "cannot store", encode, {"a": {1, 2}})
    _raises(TypeError, "cannot store", encode, {"a": object()})


def check_results():'''),
    ("def run_all():\n    check_frames()\n    check_results()", "def run_all():\n    check_frames()\n    check_codec()\n    check_results()"),
])
print("ok")
