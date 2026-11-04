def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, (p, old[:60])
        s = s.replace(old, new, 1)
    open(p, "w", encoding="utf-8").write(s)

edit("evaluation/bev_annotation_pts.py", [(
'''    print(f"{pts.n} annotations in the table;''',
'''    for tok, want in (("30f65d58c9c34488991774bec0ff0500", (3, 1)), ("5bb38a1b262649debccfd4fc00c1c600", (11, 1)), ("c28b310e5a3f4f258f1ef52e0f3c3700", (1, 2))):
        assert pts(tok) == want, (tok, pts(tok), want)         # values read from sample_annotation.json by hand: (lidar, radar) in this order; the first token ends in "00"
    print(f"{pts.n} annotations in the table;''')])

edit("evaluation/bev_task.py", [(
'''    def _pts(self):
        """token -> (lidar, radar points) of the annotation table, None when it was not built (the radar classes of `cov` are then left out)"""
        try:
            from evaluation.bev_annotation_pts import OUT, load
            return load() if os.path.exists(OUT) else None
        except Exception:                                                     # a damaged table must not stop an evaluation
            return None
''', '''    _PTS = []

    def _pts(self):
        """token -> (lidar, radar points) of the annotation table (loaded once); None when it was not built or cannot be read: the radar classes of `cov` are then left out,
        and that is said on the console"""
        if not self._PTS:
            from evaluation.bev_annotation_pts import OUT, load
            try:
                self._PTS.append(load() if os.path.exists(OUT) else None)
            except Exception as ex:                                           # a damaged table must not stop an evaluation
                print(f"WARNING: annotation table unreadable ({type(ex).__name__}: {ex})", flush=True)
                self._PTS.append(None)
            if self._PTS[0] is None:
                print(f"WARNING: no annotation table ({OUT}): `cov` leaves out the radar classes (cam_radar, cam_only, radar_only, neither); build it with `python evaluation/bev_annotation_pts.py build`", flush=True)
        return self._PTS[0]
''')])

edit("evaluation/bev_gtcheck.py", [(
'''    return "15efbde2d1" if "15efbde2d1" in have else max(have, key=have.get)''',
'''    if "15efbde2d1" in have:
        return "15efbde2d1"
    best = max(have, key=have.get)
    print(f"WARNING: no cached frames of the current GT 15efbde2d1; using {best} ({have[best]} frames)", flush=True)
    return best''')])

edit("evaluation/bev_mutants.py", [
('''    _, base = R.all_rows(n, "15efbde2d1", 0)''', '''    sig = R.G.resolve_sig(None)
    _, base = R.all_rows(n, sig, 0)'''),
('''            _, rows = R.all_rows(n, "15efbde2d1", 0)''', '''            _, rows = R.all_rows(n, sig, 0)''')])

edit("evaluation/bev_fusion_area.py", [(
'''def coverage_table(n: int = 300, sig: str = "15efbde2d1", seed: int = 0, pts=None):''', '''def coverage_table(n: int = 300, sig: Optional[str] = None, seed: int = 0, pts=None):'''),
('''    counts = []
    for i, (name, s) in enumerate(G.pick(n, sig, seed)):''', '''    counts, sig = [], G.resolve_sig(sig)
    for i, (name, s) in enumerate(G.pick(n, sig, seed)):''')])

edit("evaluation/bev_suite.py", [(
'''boxes without a lidar / radar point are dropped from the GT, a detection on one still counts as a false "
               "detection (D38 will make that two-sided).''', '''`det`: boxes without a lidar / radar point are dropped from the GT, a detection on one still counts as a false "
               "detection (`detf` / `detfp` of `eval --fusion` make that two-sided, D38).''')])

edit("evaluation/README.md", [
("neutral radius 0, detections outside the area kept", "neutral rule off, detections outside the area kept"),
("(the care threshold by the module selftest, the rest by rows)", "(the care threshold and the hidden-cells one by the module selftest, the rest by rows)"),
("182 numbers incl. the role-swapped rows", "192 numbers incl. the role-swapped rows"),
("`--selftest` (footprint shares, classes, neutral detections, coverage classes; in `tests/run_gates.sh`), the asserted `detf` / `cov` rows",
 "`--selftest` (footprint shares, corner shares, classes, neutral detections incl. a neutral box outside the window, NaN / huge boxes, coverage classes; in `tests/run_gates.sh`), the asserted `detf` / `cov` rows, the geometry rows (footprints against the cached vehicle layer; a y flip, swapped length / width or negated yaw must fall below 0.9) and the cross-check rows against `autogt/coverage.py` (same fusion cells, same corner shares). `cov` recall is a ceiling over the COMPLETE GT set (nearest first, neutral boxes included), not the recall of `detf` at 2 m"),
("check` (5298 of 5298 tokens found;", "check` (5298 of 5298 tokens found, three (lidar, radar) pairs read from the json by hand;"),
])
