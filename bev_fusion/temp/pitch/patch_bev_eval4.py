def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_eval.py", [
    ('''from evaluation import bev_suite  # noqa: E402
FINGERPRINT_BYTES = 4 << 20                      # the first and last 4 MB of a checkpoint and its size: a retrained checkpoint of the same step differs there (a whole-file hash would read 0.5 GB)
''', '''from evaluation import bev_suite  # noqa: E402

_CODE = {"path": None}                           # the code snapshot of the first network built in this process: its modules stay imported, no other snapshot can run here
'''),
    ('''    tried = []
    for cand in (text, os.path.join("runs", text)):
        p = os.path.abspath(cand if os.path.isabs(cand) else (os.path.join(ROOT, cand) if os.path.isdir(os.path.join(ROOT, cand)) else cand))
        tried.append(p)
        if os.path.isfile(os.path.join(p, "config.json")):
            return os.path.relpath(p, ROOT) if os.path.commonpath([p, ROOT]) == ROOT else p
    raise FileNotFoundError(f"{text}: no run folder with a config.json (looked at {', '.join(tried)})")''', '''    tried = [os.path.abspath(text if os.path.isabs(text) else os.path.join(ROOT, text)), os.path.abspath(os.path.join(ROOT, "runs", text))]
    for p in tried:
        if os.path.isfile(os.path.join(p, "config.json")):
            return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p
    raise FileNotFoundError(f"{text}: no run folder with a config.json (looked at {', '.join(tried)})")'''),
    ('''def identity(net: str) -> Dict[str, Any]:
    """what the predictions of this network depend on, read without loading it: the last checkpoint (name, size and a fingerprint), the config and the code snapshot"""
    net = _abs(net)
    ckpts = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint (ckpt_*.pt) in {net}")
    last = ckpts[-1]
    with open(last, "rb") as f:
        head = f.read(FINGERPRINT_BYTES)
        f.seek(max(0, os.path.getsize(last) - FINGERPRINT_BYTES))
        tail = f.read()
    with open(os.path.join(net, "config.json"), "rb") as f:
        cfg = hashlib.sha1(f.read()).hexdigest()[:12]
    try:
        with open(os.path.join(net, "code", "SNAPSHOT.txt"), encoding="utf-8") as f:
            snap = f.read().strip()
    except OSError:
        snap = None
    with open(os.path.join(ROOT, "evaluation", "bev_adapter.py"), "rb") as f:      # Subject.predict and the decoding of the outputs live there: a change of them is a change of the predictions
        adapter = hashlib.sha1(f.read().replace(b"\\r\\n", b"\\n")).hexdigest()[:12]
    return {"run": name(net), "checkpoint": os.path.basename(last), "size": os.path.getsize(last), "fingerprint": hashlib.sha1(head + tail).hexdigest()[:12], "config": cfg,
            "code": snap, "adapter": adapter}''', '''def _file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _tree_hash(folder: str) -> str:
    """the hash of the python files of a folder (name and content, line endings ignored): the code a run without a snapshot is evaluated with"""
    h = hashlib.sha1()
    for p in sorted(glob.glob(os.path.join(folder, "*.py"))):
        h.update(os.path.basename(p).encode() + hashlib.sha1(open(p, "rb").read().replace(b"\\r\\n", b"\\n")).digest())
    return h.hexdigest()[:12]


def identity(net: str) -> Dict[str, Any]:
    """what the predictions of this network depend on, read without loading it: the last checkpoint (name, size and the hash of the whole file), the config, the code (the snapshot's
    record, or a hash of the live training/ code for a run that has none: it is imported then), and Subject.predict / the output decoding in bev_adapter.py"""
    net = _abs(net)
    ckpts = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint (ckpt_*.pt) in {net}")
    last = ckpts[-1]
    try:
        with open(os.path.join(net, "code", "SNAPSHOT.txt"), encoding="utf-8") as f:
            code = f.read().strip()
    except OSError:
        code = "live training/ " + _tree_hash(os.path.join(ROOT, "training"))
    return {"run": name(net), "checkpoint": os.path.basename(last), "size": os.path.getsize(last), "sha1": _file_hash(last), "config": _file_hash(os.path.join(net, "config.json")),
            "code": code, "adapter": _file_hash(os.path.join(ROOT, "evaluation", "bev_adapter.py"))}'''),
    ('''    from evaluation import bev_sets
    bev_sets.parse(of)                                                  # TEST and malformed names are refused before anything is built
    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)''', '''    from evalsuite.api import EvaluationError
    from evaluation import bev_sets
    bev_sets.parse(of)                                                  # TEST and malformed names are refused before anything is built
    code = os.path.abspath(os.path.join(_abs(net), "code")) if os.path.isdir(os.path.join(_abs(net), "code")) else os.path.join(ROOT, "training")
    if _CODE["path"] not in (None, code):
        raise EvaluationError(f"one network per process: this process imported the code of {_CODE['path']}, {net} has {code}; run it in another process (compare the evaluations by their ids)")
    _CODE["path"] = code
    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)'''),
    ('''    ident = rs.manifest["identity"]["network"] if rs else {}
    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = _step(ident.get("checkpoint")), len(store.frames), len(store.scene_names())
    if rs is not None and set(rs.manifest.get("gts", {})) not in (set(), {info.get("gt_signature")}):       # predicted under another GT: its val loss is not this GT's
        flat = {k: v for k, v in flat.items() if not k.startswith("val_loss/")}
        better = {k: v for k, v in better.items() if not k.startswith("val_loss/")}
    fr = Frames.parse(info["frames"])
    card_ = contract_card(info["network"], fr.of, "", flat, better, len(store.frames), info.get("gt_signature"), info.get("state", "carry"), info.get("control", ""))
    if fr.kind != "all":
        card_["subset"] = str(fr)
    a = SimpleNamespace(run=info["network"], ckpt_dir=None, device="cpu", control=info.get("control", ""), split=fr.of)
    e = SimpleNamespace(n=len(store.frames), warm=True, prior=0)
    subj = {"checkpoint": ident.get("checkpoint"), "step": _step(ident.get("checkpoint")), "device": "cpu", "code": ident.get("code")}''',
     '''    ident = rs.manifest["identity"]["network"] if rs else {}
    run = _abs(info["network"])                                                # the run folder as a path that works from any working directory
    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = _step(ident.get("checkpoint")), len(store.frames), len(store.scene_names())
    gts = {fr["info"].get("gt_at_predict") for fr in store.frames if fr["info"].get("gt_at_predict")}
    if gts not in (set(), {info.get("gt_signature")}):                          # predicted under another GT: its val loss is not this GT's
        flat = {k: v for k, v in flat.items() if not k.startswith("val_loss/")}
        better = {k: v for k, v in better.items() if not k.startswith("val_loss/")}
    fr = Frames.parse(info["frames"])
    devices = sorted({fr_["info"]["device"] for fr_ in store.frames if fr_["info"].get("device")})
    card_ = contract_card(run, fr.of, "", flat, better, len(store.frames), info.get("gt_signature"), info.get("state", "carry"), info.get("control", ""))
    if fr.kind != "all":
        card_["subset"] = str(fr)
    a = SimpleNamespace(run=run, ckpt_dir=None, device=",".join(devices) or "unknown", control=info.get("control", ""), split=fr.of)
    e = SimpleNamespace(n=len(store.frames), warm=True, prior=0)
    subj = {"checkpoint": ident.get("checkpoint"), "step": _step(ident.get("checkpoint")), "device": a.device, "code": ident.get("code")}'''),
    ("provenance = ", "provenance = ") if False else ('''    from evaluation.bev_suite import better_of, contract_card, provenance_block, to_legacy''', '''    from evaluation.bev_suite import better_of, contract_card, provenance_block, to_legacy'''),
])
print("ok")
