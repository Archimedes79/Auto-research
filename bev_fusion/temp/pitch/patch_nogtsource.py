def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    ('''  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device); free_source(network, of, workers, control) and
  free_subject(network, source, device): a Source without ground truth and a Subject that only predicts, which the Runner uses (else it uses source / subject); gt_source(...): the Source
  the Evaluation scores with (else source); MEMORY_TAP = the name of a tap whose mean is the fill of the network's memory (the health check, runner_api.py).''', '''  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device); free_source(network, of, workers, control) and
  free_subject(network, source, device): a Source without ground truth and a Subject that only predicts, which the Runner uses (else it uses source / subject; the two go together);
  MEMORY_TAP = the name of a tap whose mean is the fill of the network's memory (the health check, runner_api.py).'''),
    ('''        """the Source of a set: with its ground truth (the adapter's gt_source, else source) for the Evaluation, without (free_source, else source) for the Runner. A set the project does not
        define is refused before anything is opened, and whatever the adapter raises (SystemExit included) is an EvaluationError"""''', '''        """the Source of a set: with its ground truth (the adapter's source) for the Evaluation, without (free_source, else source) for the Runner. A set the project does not define is
        refused before anything is opened, and whatever the adapter raises (SystemExit included) is an EvaluationError"""'''),
    ('''            return getattr(self.ad, "gt_source" if labelled else "free_source", self.ad.source)(net, of, workers=workers, control=control)''', '''            make = self.ad.source if labelled else getattr(self.ad, "free_source", self.ad.source)
            return make(net, of, workers=workers, control=control)'''),
    ('''        source = self._source(info["network"], fr.of, workers, info.get("control", ""))
        suite = EvalSuite(''', '''        source = self._source(info["network"], fr.of, workers, info.get("control", ""), True)
        suite = EvalSuite('''),
])
patch("evalsuite/README.md", [
    ("   optional: free_source free_subject gt_source reference card time probe resolve release inputs MEMORY_TAP COMPARE_IGNORE", "   optional: free_source free_subject reference card time probe resolve release inputs MEMORY_TAP COMPARE_IGNORE"),
])
patch("../bev_fusion/evaluation/bev_eval.py", [
    ('''_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here
_SOURCES = {}                                    # (run, set, workers, control) -> its Source while it holds its Subject: the Evaluation and the Runner ask for the same one in a row
''', '''_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here
'''),
    ('''    _CODE["path"] = code
    key = (net, of, workers, control)
    if key in _SOURCES and getattr(_SOURCES[key], "_bev_subject", None) is not None:
        return _SOURCES[key]                                            # built a moment ago for the same question: the model is loaded once
    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)
    src._bev_subject = subj
    src._bev_key = key
    _SOURCES[key] = src
    return src''', '''    _CODE["path"] = code
    src, subj, _ = _A().build(net, device="cpu", split=of, workers=workers, control=control)
    src._bev_subject = subj
    return src'''),
    ('''    src._bev_subject = None
    _SOURCES.pop(getattr(src, "_bev_key", None), None)
    import gc
    gc.collect()''', '''    src._bev_subject = None
    import gc
    gc.collect()'''),
    ('''    built._bev_subject = None                                           # the free Source holds it now; the labelled one is only a reference reader from here on
    _SOURCES.pop(getattr(built, "_bev_key", None), None)
    return free''', '''    built._bev_subject = None                                           # the free Source holds it now
    return free'''),
])
print("ok")
