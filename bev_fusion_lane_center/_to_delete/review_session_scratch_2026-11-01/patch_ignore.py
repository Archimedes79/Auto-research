import sys
q, st, bs = sys.argv[1:4]
def patch(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8", newline="").write(s)
patch(q, [
('''    def compare(self, run_a: Union[str, RunStore], run_b: Union[str, RunStore], boot: int = 1000, seed: int = 0, force: bool = False
                ) -> Dict[str, Any]:''', '''    def compare(self, run_a: Union[str, RunStore], run_b: Union[str, RunStore], boot: int = 1000, seed: int = 0, force: bool = False,
                ignore: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Any]:'''),
('''        version, caches); otherwise the counts are of different questions and the comparison is refused (force=True compares anyway)."""
        a, b = self.open(run_a), self.open(run_b)
        same_question(a, b, force=force)''', '''        version, caches); otherwise the counts are of different questions and the comparison is refused (force=True compares anyway).
        ignore: description keys that are the arms' difference, not the question ({"source": ("feature_cache",)}: a run fed with
        other features answers the same question on the same frames and ground truth)."""
        a, b = self.open(run_a), self.open(run_b)
        same_question(a, b, force=force, ignore=ignore)'''),
('''def same_question(*stores: RunStore, force: bool = False) -> None:
    """refuse runs whose manifests differ in `source` or `task` (dataset, split, ground-truth version, caches: the bev_fusion GT signature
    trap) unless force=True; the first run is the reference"""''', '''def same_question(*stores: RunStore, force: bool = False, ignore: Optional[Dict[str, Sequence[str]]] = None) -> None:
    """refuse runs whose manifests differ in `source` or `task` (dataset, split, ground-truth version, caches: the bev_fusion GT signature
    trap) unless force=True; the first run is the reference. ignore: {entry: keys} that may differ (the inputs the arms vary)"""'''),
('''        diff = _identity_differences(stores[0].manifest, st.manifest, ("source", "task"), gained=gained, either=True)''',
 '''        diff = _identity_differences(stores[0].manifest, st.manifest, ("source", "task"), gained=gained, either=True, ignore=ignore)'''),
])
patch(st, [
('''def _identity_differences(stored: Dict[str, Any], new: Dict[str, Any], keys=_IDENTITY_KEYS, gained: Optional[List[str]] = None,
                          either: bool = False) -> List[str]:''', '''def _identity_differences(stored: Dict[str, Any], new: Dict[str, Any], keys=_IDENTITY_KEYS, gained: Optional[List[str]] = None,
                          either: bool = False, ignore: Optional[Dict[str, Any]] = None) -> List[str]:'''),
('''    one side has, in both directions. A key present on both sides with another value always differs."""''',
 '''    one side has, in both directions. A key present on both sides with another value always differs, unless `ignore` names it
    ({entry: keys}: inputs that two compared arms vary on purpose, e.g. a feature cache)."""'''),
('''            drop = _VOLATILE.get(k, ())''', '''            drop = tuple(_VOLATILE.get(k, ())) + tuple((ignore or {}).get(k, ()))'''),
])
patch(bs, [
('''    res = q.compare(os.path.join(c.run_a, "eval", "suite", "quant", name), os.path.join(c.run_b, "eval", "suite", "quant", name), c.boot, c.seed)''',
 '''    res = q.compare(os.path.join(c.run_a, "eval", "suite", "quant", name), os.path.join(c.run_b, "eval", "suite", "quant", name), c.boot, c.seed,
                    force=c.force, ignore={"source": ("feature_cache", "control")})   # the arms may differ in what the model is fed (t8: features)'''),
])
print("ok")
