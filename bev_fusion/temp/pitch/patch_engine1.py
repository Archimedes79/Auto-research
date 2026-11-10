def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/quantitative.py", [
    ('''            name: Optional[str] = None, resume: bool = True, progress: int = 25, on_frame=None,
            ignore: Optional[Dict[str, Sequence[str]]] = None) -> RunStore:''',
     '''            name: Optional[str] = None, resume: bool = True, progress: int = 25, on_frame=None,
            ignore: Optional[Dict[str, Sequence[str]]] = None, results=None, reference=None) -> RunStore:'''),
    ('''        resumed store that may differ ({"source": ("evaluator_hash",)}: the person accepts a store finished with other evaluator code).
        -> the closed RunStore."""''',
     '''        resumed store that may differ ({"source": ("evaluator_hash",)}: the person accepts a store finished with other evaluator code).
        results: a ResultSet (results.py): score ITS stored predictions instead of running the Subject (the frames are its frames, or `frames` out of them; no network, no state to
        carry, no plausibility flags or probe: they need the inputs). reference(ref) -> the ground truth of a frame (default: `source.load(ref).reference`, which loads everything).
        -> the closed RunStore."""'''),
    ('''        if frames is not None:
            by: Dict[str, List[FrameRef]] = {}''',
     '''        if results is not None and frames is None:
            frames = results.refs()
        if frames is not None:
            by: Dict[str, List[FrameRef]] = {}'''),
    ('''            if state == "carry":
                for sc, fr in plan.items():''', '''            if state == "carry" and results is None:
                for sc, fr in plan.items():'''),
    ('''        if "ops" not in cost:
            ops = self._ops(''', '''        if "ops" not in cost and results is None:
            ops = self._ops('''),
    ('''        runner = Runner(src, subj, state)
        t0, n = time.perf_counter(), 0''', '''        runner = Runner(src, subj, state) if results is None else None
        ref_of = reference or (lambda ref: src.load(ref).reference)
        t0, n = time.perf_counter(), 0'''),
    ('''                for sample, pred in runner.iter_scene(sc, refs):
                    n += 1''', '''                stream = runner.iter_scene(sc, refs) if results is None else (
                    (Sample(ref, None, ref_of(ref), meta={"token": ref.key}), results.load(ref)) for ref in refs if ref.label not in done)
                for sample, pred in stream:
                    n += 1'''),
    ('''                    flags = [dict(name=f.name, status=f.status, value=_plain(f.value), note=f.note)
                             for f in task.checks(sample, pred) if f.status != "ok"]''',
     '''                    flags = [dict(name=f.name, status=f.status, value=_plain(f.value), note=f.note)
                             for f in task.checks(sample, pred) if f.status != "ok"] if results is None else []'''),
    ('''                    pr = task.probe(sample, pred)
                    if pr:''', '''                    pr = task.probe(sample, pred) if results is None else None
                    if pr:'''),
])
print("ok")
