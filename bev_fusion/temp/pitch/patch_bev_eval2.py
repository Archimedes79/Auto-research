def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_eval.py", [
    ('''_BUILT: Dict[int, Any] = {}                      # id(source) -> its Subject: build() makes both, the interface asks for them one after the other


def _A():''', '''
def _abs(net: str) -> str:
    return net if os.path.isabs(net) else os.path.join(ROOT, net)


def resolve(text: str) -> str:
    """a run name (`t23_C_s1`), a path relative to bev_fusion (`runs/t23_C_s1`) or an absolute one -> the run folder, as a path relative to bev_fusion when it lies inside it"""
    tried = []
    for cand in (text, os.path.join("runs", text)):
        p = os.path.abspath(cand if os.path.isabs(cand) else (os.path.join(ROOT, cand) if os.path.isdir(os.path.join(ROOT, cand)) else cand))
        tried.append(p)
        if os.path.isfile(os.path.join(p, "config.json")):
            return os.path.relpath(p, ROOT) if os.path.commonpath([p, ROOT]) == ROOT else p
    raise FileNotFoundError(f"{text}: no run folder with a config.json (looked at {', '.join(tried)})")


def _A():'''),
    ('''def out(net: str) -> str:
    return os.path.join(net, "eval")''', '''def out(net: str) -> str:
    return os.path.join(_abs(net), "eval")'''),
    ('''    ckpts = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint (ckpt_*.pt) in {net}")''', '''    net = _abs(net)
    ckpts = sorted(glob.glob(os.path.join(net, "ckpt_*.pt")))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint (ckpt_*.pt) in {net}")'''),
    ('''    return {"run": name(net), "checkpoint": os.path.basename(last), "size": os.path.getsize(last), "fingerprint": hashlib.sha1(head + tail).hexdigest()[:12], "config": cfg, "code": snap}''',
     '''    with open(os.path.join(ROOT, "evaluation", "bev_adapter.py"), "rb") as f:      # Subject.predict and the decoding of the outputs live there: a change of them is a change of the predictions
        adapter = hashlib.sha1(f.read().replace(b"\\r\\n", b"\\n")).hexdigest()[:12]
    return {"run": name(net), "checkpoint": os.path.basename(last), "size": os.path.getsize(last), "fingerprint": hashlib.sha1(head + tail).hexdigest()[:12], "config": cfg,
            "code": snap, "adapter": adapter}'''),
    ('''    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)
    _BUILT[id(src)] = subj
    return src


def subject(net: str, src, device: str):
    subj = _BUILT.pop(id(src))
    if str(device).startswith("cuda") and not str(subj.device).startswith("cuda"):
        import torch
        if torch.cuda.is_available():
            subj.model.to(device)
            subj.loss.to(device)
            subj.device = device
    return subj''', '''    from evaluation import bev_sets
    bev_sets.parse(of)                                                  # TEST and malformed names are refused before anything is built
    A = _A()
    src, subj, _ = A.build(net, device="cpu", split=of, workers=workers, control=control)
    src._bev_subject = subj
    return src


def subject(net: str, src, device: str):
    """the Subject `source` built with it, on `device` (a model built on the CPU is moved; cuda asked for and not there is an error, never a quiet CPU run)"""
    subj = getattr(src, "_bev_subject", None)
    if subj is None:
        raise RuntimeError("the Subject of this source was handed out or released already")
    if str(device).startswith("cuda") and not str(subj.device).startswith("cuda"):
        import torch
        from evalsuite.api import EvaluationError
        if not torch.cuda.is_available():
            raise EvaluationError(f"device {device} was asked for and CUDA is not available (pass device=cpu to run on the CPU)")
        subj.model.to(device)
        subj.loss.to(device)
        subj.device = device
    src._bev_subject = None
    return subj


def release(src) -> None:
    """drop the Subject a source holds (a check or a rescoring never runs it)"""
    src._bev_subject = None'''),
    ('''    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = _step(ident.get("checkpoint")), len(store.frames), len(store.scene_names())''',
     '''    flat["info/step"], flat["info/n_samples"], flat["info/n_scenes"] = _step(ident.get("checkpoint")), len(store.frames), len(store.scene_names())
    if rs is not None and set(rs.manifest.get("gts", {})) not in (set(), {info.get("gt_signature")}):       # predicted under another GT: its val loss is not this GT's
        flat = {k: v for k, v in flat.items() if not k.startswith("val_loss/")}
        better = {k: v for k, v in better.items() if not k.startswith("val_loss/")}'''),
    ('''    card_["provenance"] = provenance_block(a, e, store, subj, info.get("gt_signature"), "", info.get("state", "carry"), start={}, avail=store.scene_names())
    return card_''', '''    card_["provenance"] = provenance_block(a, e, store, subj, info.get("gt_signature"), "", info.get("state", "carry"), start={}, avail=store.scene_names())
    card_["provenance"]["launcher"] = "evalsuite Evaluation.run: a direct call, no run.py (no ledger row, no GPU slot): a diagnostic unless it was started through run.py"
    return card_'''),
    ('''        return bev_timing.run_timing(net, n=n)''', '''        return bev_timing.run_timing(_abs(net), n=n)'''),
    ('''    return bev_modules.probe(net, frames=frames, device=device)''', '''    return bev_modules.probe(_abs(net), frames=frames, device=device)'''),
    ('''COMPARE_IGNORE = {"source": ("feature_cache", "control", "evaluator_hash")}       # the arms may differ in what the model is fed; the evaluator hash only guards a resume''',
     '''COMPARE_IGNORE = bev_suite.COMPARE_IGNORE                                    # the arms may differ in what the model is fed; the evaluator hash only guards a resume'''),
    ('''from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))''', '''from typing import Any, Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from evaluation import bev_suite  # noqa: E402'''),
    ("import re\nfrom types import SimpleNamespace", "import re\nimport sys\nfrom types import SimpleNamespace"),
])
patch("../autoresearch/evalsuite/api.py", [
    ('''    def _net(self, network: str) -> str:
        net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        self._networks[self.ad.name(net)] = net
        return net''', '''    def _net(self, network: str) -> str:
        try:
            net = self.ad.resolve(network) if hasattr(self.ad, "resolve") else network
        except OSError as e:
            raise EvaluationError(str(e)) from None
        self._networks[self.ad.name(net)] = net
        return net'''),
])
print("ok")
