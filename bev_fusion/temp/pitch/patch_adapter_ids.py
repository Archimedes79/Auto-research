def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("../bev_fusion/evaluation/bev_eval.py", [
    ('''def _tree_hash(folder: str) -> str:
    """the hash of the python files of a folder (name and content, line endings ignored): the code a run without a snapshot is evaluated with"""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(folder, "*.py"))):
        h.update(os.path.basename(p).encode() + hashlib.sha1(open(p, "rb").read().replace(b"\\r\\n", b"\\n")).digest())
    return h.hexdigest()[:12]''', '''def _tree_hash(folder: str) -> str:
    """the hash of the python files of a folder (name and content, line endings ignored): the code a run without a snapshot is evaluated with"""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(folder, "*.py"))):
        with open(p, "rb") as f:
            h.update(os.path.basename(p).encode() + hashlib.sha1(f.read().replace(b"\\r\\n", b"\\n")).digest())
    return h.hexdigest()[:12]'''),
    ('''    """what the predictions of this network depend on, read without loading it: the last checkpoint (name, size and the hash of the whole file), the config, the code (the snapshot's
    record, or a hash of the live training/ code for a run that has none: it is imported then), and Subject.predict / the output decoding in bev_adapter.py"""''', '''    """what the predictions of this network depend on, read without loading it: the last checkpoint (name and the hash of the whole file), the config, the code (the snapshot's record, or a
    hash of the live training/ code for a run that has none: it is imported then), and Subject.predict / the output decoding in bev_adapter.py and bev_free.py (the two paths give the same predictions)"""'''),
    ('''    return {"run": name(net), "checkpoint": os.path.basename(last), "size": os.path.getsize(last), "sha1": _file_hash(last), "config": _file_hash(os.path.join(net, "config.json")),
            "code": code, "adapter": _file_hash(os.path.join(ROOT, "evaluation", "bev_adapter.py"))}''', '''    adapter = hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in ("bev_adapter.py", "bev_free.py")).encode()).hexdigest()[:12]
    return {"run": name(net), "checkpoint": os.path.basename(last), "sha1": _file_hash(last), "config": _file_hash(os.path.join(net, "config.json")), "code": code, "adapter": adapter}'''),
    ('''    from evaluation.bev_suite import better_of, contract_card, provenance_block, to_legacy
    from evalsuite.api import Frames
    tab = quant.table(store)''', '''    from evaluation import bev_adapter
    from evaluation.bev_suite import better_of, contract_card, provenance_block, to_legacy
    from evalsuite.api import EvaluationError, Frames
    if info.get("evaluator") and info["evaluator"] != bev_adapter.evaluator_hash():
        raise EvaluationError(f"{info['id']} was scored with other evaluator code than the one now ({info['evaluator']} / {bev_adapter.evaluator_hash()}): score the predictions again "
                              f"(`score {info['results']}`, no network is run), then make the card of the new evaluation")
    tab = quant.table(store)'''),
])
patch("evalsuite/runner_api.py", [
    ('''                self._memory_health(rs, sc, fill)''', '''                self._memory_health(rs, sc, fill)
                rs.save()                                                      # a killed writer keeps what it recorded so far (frames are atomic, the manifest per scene)'''),
])
print("ok")
