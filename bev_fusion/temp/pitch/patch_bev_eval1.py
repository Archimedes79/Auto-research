def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_sets.py", [
    ('''def set_hash(name: str, splits_dir: str = None) -> str:''', '''def counts(splits_dir: str = None) -> dict:
    """{set: number of scenes} of every set the splits folder defines"""
    S = _splits()
    out = {}
    for n in names(splits_dir):
        split, part = n.split(".")
        out[n] = len(S.load(split, *([splits_dir] if splits_dir else []))[part])
    return out


def set_hash(name: str, splits_dir: str = None) -> str:'''),
    ('''    assert "tiny.val" in table(tmp) and "toy" in table(tmp)''', '''    assert "tiny.val" in table(tmp) and "toy" in table(tmp) and counts(tmp) == {"tiny.train": 2, "tiny.val": 1} and counts()["small.val"] == 10'''),
])
patch("evaluation/bev_eval.py", [
    ('''    from evaluation import bev_sets
    S = bev_sets._splits()
    res: Dict[str, Any] = {"val": None, "train": None}
    for n in bev_sets.names():
        split, part = n.split(".")
        res[n] = len(S.load(split)[part])
    return res''', '''    from evaluation import bev_sets
    return {"val": None, "train": None, **bev_sets.counts()}'''),
    ('''    from evaluation import bev_suite
    from evaluation.bev_suite import better_of''', '''    from evaluation.bev_suite import better_of'''),
])
patch("../autoresearch/evalsuite/api.py", [
    ('''    def _identity(self, network: str, source: Source) -> Dict[str, Any]:
        inputs = {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        return {"network": self.ad.identity(network), "inputs": inputs, "state": "carry"}''', '''    def _identity(self, network: str, source: Source) -> Dict[str, Any]:
        inputs = {k: v for k, v in source.describe().items() if k not in RESULT_DROP}
        try:
            ident = self.ad.identity(network)
        except OSError as e:
            raise EvaluationError(f"{network}: {e}") from None
        return {"network": ident, "inputs": inputs, "state": "carry"}'''),
])
open("evaluation/evalsuite.json", "w", encoding="utf-8").write('{"adapter": "evaluation.bev_eval"}\n')
print("ok")
