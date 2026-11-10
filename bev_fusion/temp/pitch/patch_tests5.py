def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/results.py", [
    ('''FORMAT = 2
''', '''FORMAT = 2
FRAME_FILE = re.compile(r"(.+)_(\\d+)\\.npz")                             # <scene>_<index>.npz: a frame (anything else in frames/ is not)
'''),
    ('''            self._have = {f for f in os.listdir(folder) if f.endswith(".npz")} if os.path.isdir(folder) else set()''', '''            self._have = {f for f in os.listdir(folder) if FRAME_FILE.fullmatch(f)} if os.path.isdir(folder) else set()'''),
    ('''        out = []
        for f in self._listing():
            m = re.fullmatch(r"(.+)_(\\d+)\\.npz", f)
            if m:
                out.append(FrameRef(m.group(1), int(m.group(2))))
        return sorted(out)''', '''        return sorted(FrameRef(m.group(1), int(m.group(2))) for m in map(FRAME_FILE.fullmatch, self._listing()))'''),
    ('''        """the stored frames (scene and index; the dataset key is in the file, not in its name), in time order; a file not named <scene>_<index>.npz is not a frame"""''', '''        """the stored frames (scene and index; the dataset key is in the file, not in its name), in time order"""'''),
])

patch("evalsuite/selftest_api.py", [
    # ---- toy: an identity that can change while the network is loaded
    ('''STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok")''', '''STATE = SimpleNamespace(root="", gt="g1", shift=0, calls=0, loads=0, free_loads=0, fail_at=None, memory="ok", bump=0, change=False)'''),
    ('''def identity(net): return {"network": net}''', '''def identity(net): return {"network": net, "checkpoint": STATE.bump}'''),
    ('''free_subject = subject                                                               # (the toy network never reads a ground truth)''', '''def free_subject(net, src, device):                                                  # (the toy network never reads a ground truth)
    STATE.bump += 1 if STATE.change else 0                                           # "a training wrote a new checkpoint while the network was loaded"
    return subject(net, src, device)'''),
    # ---- results: a foreign file in frames/, a damaged frame, a folder that package.py clean emptied
    ('''    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.npz"), "wb") as f:
        f.write(b"not a frame")
    _raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[0])''', '''    with open(os.path.join(tmp, rid, "frames", f"{refs[0].slug}.npz"), "wb") as f:
        f.write(b"not a frame")
    _raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[0])
    good = open(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), "rb").read()
    mid = bytearray(good)
    mid[len(mid) // 2] ^= 0xFF                                                          # a byte in the middle of the compressed data: zlib / CRC error, not a traceback
    with open(os.path.join(tmp, rid, "frames", f"{refs[1].slug}.npz"), "wb") as f:
        f.write(bytes(mid))
    _raises(IOError, "is unreadable", ResultSet.open(os.path.join(tmp, rid)).load, refs[1])
    with open(os.path.join(tmp, rid, "frames", "notes.npz"), "wb") as f:                # a file that is not named <scene>_<index>.npz is not a frame
        f.write(b"x")
    assert [r.label for r in ResultSet.open(os.path.join(tmp, rid)).refs()] == [r.label for r in refs] and ResultSet.open(os.path.join(tmp, rid)).n_frames() == 3 + 1
    # package.py clean removes the emptied frames / taps folders and keeps the manifest: the set takes frames again
    shutil.rmtree(os.path.join(tmp, rid, "frames"))
    shutil.rmtree(os.path.join(tmp, rid, "taps"))
    empty = ResultSet.open(os.path.join(tmp, rid), "a")
    assert empty.n_frames() == 0 and not empty.has(refs[0])
    empty.add(refs[0], Prediction(outputs="o", taps={"t": np.ones(1)}, info={}))
    assert empty.has(refs[0], ("t",)) and empty.load(refs[0], ("t",)).outputs == "o"'''),
    # ---- the adapter pair, bare-string taps, the identity changing while the network is loaded, the health at its boundaries
    ('''    # intermediate results: asked for with the predictions; a frame stored without them is predicted again, one with them is reused; unknown taps are refused before the network runs''', '''    # the adapter: free_source and free_subject go together
    half = SimpleNamespace(**{f: getattr(ME, f) for f in ADAPTER_FUNCTIONS}, free_source=free_source)
    _raises(EvaluationError, "go together", load_adapter, half, None)
    _raises(EvaluationError, "go together", Evaluation, None, half)
    # the identity of the network changes while it is loaded (a training wrote a checkpoint): nothing is stored, the message says why
    STATE.change = True
    try:
        _raises(EvaluationError, "changed while it was loaded", ev.runner.run, "runs/thr0.8", "scenes:demo-0:val")
    finally:
        STATE.change, STATE.bump = False, 0
    assert [x for x in ev.list("runs/thr0.8")] == [], "nothing was stored"
    # intermediate results: asked for with the predictions; a frame stored without them is predicted again, one with them is reused; unknown taps are refused before the network runs'''),
    ('''    _raises(EvaluationError, "unknown taps", ev.runner.run, net5, "val", taps=("nothing",))''', '''    _raises(EvaluationError, "unknown taps", ev.runner.run, net5, "val", taps=("nothing",))
    assert ev.runner.run(net5, "scenes:demo-0:val", taps="feat")["frames_predicted"] == 0 and ev.runner.check(net5, "scenes:demo-0:val", taps="feat")["frames_stored"] == 6, "a bare string is one tap, not its letters"'''),
    ('''    # the dry run: what would happen, and the problems before anything is run''', '''    # the health at its boundaries (D70): 3 frames are too few to judge the filling; a start of exactly 0.05 and a growth of exactly 0.02 are healthy
    def health(fill):
        rs_ = SimpleNamespace(manifest={})
        Runner._memory_health(rs_, "s", fill)
        return rs_.manifest["health"]
    assert health([0.0, 0.0, 0.0]) == {} and health([0.05, 0.1, 0.2, 0.3]) == {} and health([0.0, 0.02, 0.0, 0.0]) == {} and health([]) == {}
    assert "not empty" in health([0.06, 0.2, 0.4, 0.6])["s"] and "does not fill" in health([0.0, 0.01, 0.0, 0.01])["s"]
    # the dry run: what would happen, and the problems before anything is run'''),
    ('''from .api import Evaluation, EvaluationError, Frames''', '''from .api import ADAPTER_FUNCTIONS, Evaluation, EvaluationError, Frames, load_adapter'''),
])
print("ok")
