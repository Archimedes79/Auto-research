def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/results.py", [
    ('''Layout (`format` 1 in the manifest): `<path>/manifest.json`, `<path>/frames/<scene>_<index>.pkl` (a zlib level 1 compressed pickle of {outputs, info, key}: the program's own cache, not an
exchange format, `pickle.load` cannot read it directly, use `load`; open a ResultSet only from a folder this program made) and, for the intermediate results asked for with the predictions
(taps), `<path>/taps/<scene>_<index>/<tap>.npy`. A frame stored without a tap does not have it: `has(ref, taps)` says no, and the frame is predicted again to get it.''',
     '''Layout (`format` 2 in the manifest): `<path>/manifest.json`, `<path>/frames/<scene>_<index>.npz` and, for the intermediate results asked for with the predictions (taps),
`<path>/taps/<scene>_<index>/<tap>.npy`. A frame file is an ordinary numpy .npz (zlib level 1; `np.load(file)` reads it, no pickle anywhere): the arrays of the prediction as `a0.npy`, `a1.npy`,
... and `tree.json`, the nested outputs / info / key as json in which an array is `{"$a": i}`, a dict is `{"$d": [[key, value], ...]}` (keys keep their type: 0.0 is a float key), a tuple
`{"$t": [...]}`, a numpy scalar `{"$s": i}` and a float that is not finite `{"$f": "nan"}`. A frame stored without a tap does not have it: `has(ref, taps)` says no, and the frame is predicted
again to get it.'''),
    ("import os\nimport pickle\nimport socket\nimport time\nimport uuid\nimport zlib\n", "import io\nimport os\nimport socket\nimport time\nimport uuid\nimport zipfile\n"),
    ("FORMAT = 1\n", '''FORMAT = 2


def encode(obj: Any):
    """(tree, arrays): obj as json-able data and the list of its numpy arrays (see the module text for the grammar); a type that is not stored raises TypeError, an object array too"""
    arrays: List[np.ndarray] = []

    def enc(o):
        if o is None or isinstance(o, (bool, str)):
            return o
        if isinstance(o, int):
            return int(o)
        if isinstance(o, float):
            return float(o) if np.isfinite(o) else {"$f": repr(float(o))}
        if isinstance(o, np.ndarray):
            if o.dtype == object:
                raise TypeError("an object array cannot be stored")
            arrays.append(np.ascontiguousarray(o))
            return {"$a": len(arrays) - 1}
        if isinstance(o, np.generic):
            arrays.append(np.asarray(o))
            return {"$s": len(arrays) - 1}
        if isinstance(o, dict):
            return {"$d": [[enc(k), enc(v)] for k, v in o.items()]}
        if isinstance(o, tuple):
            return {"$t": [enc(x) for x in o]}
        if isinstance(o, list):
            return [enc(x) for x in o]
        raise TypeError(f"cannot store a {type(o).__name__}")
    return enc(obj), arrays


def decode(tree: Any, arrays: Sequence[np.ndarray]) -> Any:
    """the inverse of encode"""
    def dec(o):
        if isinstance(o, list):
            return [dec(x) for x in o]
        if isinstance(o, dict):
            (tag, val), = o.items()
            if tag == "$a":
                return arrays[val]
            if tag == "$s":
                return arrays[val][()]
            if tag == "$d":
                return {dec(k): dec(v) for k, v in val}
            if tag == "$t":
                return tuple(dec(x) for x in val)
            if tag == "$f":
                return float(val)
            raise ValueError(f"unknown tag {tag}")
        return o
    return dec(tree)
'''),
    ('''    def _file(self, ref: FrameRef) -> str:
        return os.path.join(self.path, "frames", f"{ref.slug}.pkl")''', '''    def _file(self, ref: FrameRef) -> str:
        return os.path.join(self.path, "frames", f"{ref.slug}.npz")'''),
    ('''            self._have = {f for f in os.listdir(folder) if f.endswith(".pkl")} if os.path.isdir(folder) else set()''', '''            self._have = {f for f in os.listdir(folder) if f.endswith(".npz")} if os.path.isdir(folder) else set()'''),
    ('''        try:
            with open(self._file(ref), "rb") as f:
                d = pickle.loads(zlib.decompress(f.read()))
            arrays = {t: np.load(self._tap_file(ref, t)) for t in taps}
            outputs, info, key = d["outputs"], d["info"], d.get("key", "")
        except (OSError, ValueError, KeyError, AttributeError, ImportError, zlib.error, pickle.UnpicklingError, EOFError) as e:
            raise IOError(f"{self._file(ref)} (or a tap of it) is unreadable ({type(e).__name__}: {e}); remove the file, the next run predicts the frame again") from None''', '''        try:
            with zipfile.ZipFile(self._file(ref)) as z:
                tree = json.loads(z.read("tree.json").decode("utf-8"))
                arrs = [np.lib.format.read_array(io.BytesIO(z.read(f"a{i}.npy")), allow_pickle=False) for i in range(len(z.namelist()) - 1)]
            d = decode(tree, arrs)
            arrays = {t: np.load(self._tap_file(ref, t), allow_pickle=False) for t in taps}
            outputs, info, key = d["outputs"], d["info"], d.get("key", "")
        except (OSError, ValueError, KeyError, IndexError, EOFError, zipfile.BadZipFile) as e:
            raise IOError(f"{self._file(ref)} (or a tap of it) is unreadable ({type(e).__name__}: {e}); remove the file, the next run predicts the frame again") from None'''),
    ('''        if not self.has(ref):
            _write_atomic(self._file(ref), zlib.compress(pickle.dumps({"outputs": pred.outputs, "info": pred.info, "key": ref.key}, protocol=4), 1))
            self._listing().add(os.path.basename(self._file(ref)))''', '''        if not self.has(ref):
            tree, arrays = encode({"outputs": pred.outputs, "info": pred.info, "key": ref.key})
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
                z.writestr("tree.json", json.dumps(tree).encode("utf-8"))
                for i, a in enumerate(arrays):
                    with z.open(f"a{i}.npy", "w", force_zip64=True) as f:
                        np.lib.format.write_array(f, a, allow_pickle=False)
            _write_atomic(self._file(ref), buf.getvalue())
            self._listing().add(os.path.basename(self._file(ref)))'''),
])
print("ok")
