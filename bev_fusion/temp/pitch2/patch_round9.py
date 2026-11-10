def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/api.py", [
    # ---- 1 the card is a file a decision can read
    ('''    def make_card(self, id: str) -> Dict[str, Any]:
        """the metric card in the project's contract (what its decision tool reads); refused when the checkpoint changed since the predictions"""
        if not hasattr(self.ad, "card"):
            raise EvaluationError("this project's adapter defines no card")
        info, store, q = self._open(id)
        rs = self._predictions(info)
        self._same_network(info, rs)
        return self.ad.card(info, store, q, rs)''', '''    def make_card(self, id: str, out: str = "") -> Dict[str, Any]:
        """the metric card in the project's contract, WRITTEN as a file (what the decision tool reads: name it in the experiment card's `arm.cards`; `python autoresearch/tools/card.py check FILE`
        checks it): `out` = the file (a folder: the file `card_<e-id>.json` in it), default `<the network's eval folder>/card_<e-id>.json`, atomic. -> {"path", "card"}. Refused when the
        checkpoint changed since the predictions"""
        if not hasattr(self.ad, "card"):
            raise EvaluationError("this project's adapter defines no card")
        info, store, q = self._open(id)
        rs = self._predictions(info)
        self._same_network(info, rs)
        card = self.ad.card(info, store, q, rs)
        fname = f"card_{str(id).partition('@')[2]}.json"
        path = os.path.abspath(out) if out else os.path.join(self.ad.out(info["network"]), fname)
        if os.path.isdir(path):
            path = os.path.join(path, fname)
        write_atomic(path, json.dumps(card, indent=1, default=str).encode("utf-8"))
        return {"path": path, "card": card}'''),
    ("from .results import Locked, ResultSet, digest, writer_lock", "from .results import Locked, ResultSet, digest, write_atomic, writer_lock"),
    # ---- 2 an evaluator that cannot be hashed fails closed
    ('''            if not desc.get("gt_signature"):
                problems.append("the data has no ground-truth signature: the numbers could not be told from another GT's")
            self._probe_reference''', '''            if not desc.get("gt_signature"):
                problems.append("the data has no ground-truth signature: the numbers could not be told from another GT's")
            if not desc.get("evaluator_hash"):
                problems.append(NO_EVALUATOR_HASH)
            self._probe_reference'''),
    ('''    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> None:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")''', '''    def _need_gt(desc: Dict[str, Any], gt_key: Optional[str]) -> None:
        if not desc.get("gt_signature"):
            raise EvaluationError("the data has no ground-truth signature: numbers could not be told from those of another GT; fix the data adapter")
        if not desc.get("evaluator_hash"):
            raise EvaluationError(NO_EVALUATOR_HASH)'''),
    ('''RESULT_DROP = ''', '''NO_EVALUATOR_HASH = ("the data adapter gives no evaluator hash (the files that shape the numbers could not be hashed: one is missing or renamed): an old evaluation would be reused across edits of the metric "
                     "code; fix the adapter")
RESULT_DROP = '''),
    ('''        now = self.ad.evaluator_now() if hasattr(self.ad, "evaluator_now") else None
        if now and info.get("evaluator") and now != info["evaluator"]:
            warn.append(''', '''        now = self.ad.evaluator_now() if hasattr(self.ad, "evaluator_now") else info.get("evaluator")
        if info.get("evaluator") and not now:
            warn.append("the evaluator code cannot be hashed now (a file that shapes the numbers is missing or renamed): whether it changed since this evaluation cannot be told")
        elif now and info.get("evaluator") and now != info["evaluator"]:
            warn.append('''),
])
patch("evalsuite/results.py", [
    ("def _write_atomic(path: str, data: bytes) -> None:", "def write_atomic(path: str, data: bytes) -> None:"),
    ("_write_atomic(", "write_atomic("),
]) if False else None

# ------------------------------------------------------------------------------------------------ results.py: the atomic writer is public (one helper for frames, taps, manifests, cards)
s = open("evalsuite/results.py", encoding="utf-8").read()
assert s.count("_write_atomic") >= 4
s = s.replace("_write_atomic", "write_atomic")
open("evalsuite/results.py", "w", encoding="utf-8").write(s)

# ------------------------------------------------------------------------------------------------ the BEV adapter
patch("../bev_fusion/evaluation/bev_eval.py", [
    ('''    adapter = hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in ("bev_adapter.py", "bev_free.py")).encode()).hexdigest()[:12]''', '''    adapter = hashlib.sha1("".join(_file_hash(p) for p in _PREDICTION_FILES).encode()).hexdigest()[:12]'''),
    ('''def _claim_code(net: str) -> str:''', '''_PREDICTION_FILES = (os.path.join(ROOT, "evaluation", "bev_adapter.py"), os.path.join(ROOT, "evaluation", "bev_free.py"),
                     os.path.join(ROOT, "..", "autoresearch", "evalsuite", "runner.py"), os.path.join(ROOT, "..", "autoresearch", "evalsuite", "torch_taps.py"))       # the code that makes a prediction: the output
# decoding of both paths, the frame loop (memory carry / reset) and the taps; part of the identity of the predictions, so an edit of any of them predicts again


def _claim_code(net: str) -> str:'''),
    ('''    if info.get("evaluator") and info["evaluator"] != bev_adapter.evaluator_hash():
        raise EvaluationError(f"{info['id']} was scored with other evaluator code than the one now ({info['evaluator']} / {bev_adapter.evaluator_hash()}): score the predictions again "
                              f"(`score {info['results']}`, no network is run), then make the card of the new evaluation")''', '''    now = bev_adapter.evaluator_hash()
    if info.get("evaluator") != now or not now:                                  # a hash that cannot be computed is not "unchanged"
        raise EvaluationError(f"{info['id']} was scored with other evaluator code than the one now ({info.get('evaluator')} / {now}; None = a file that shapes the numbers is missing or renamed): "
                              f"score the predictions again (`score {info['results']}`, no network is run), then make the card of the new evaluation")'''),
    ("    ident = rs.manifest[\"identity\"][\"network\"] if rs else {}\n", "    ident = rs.manifest[\"identity\"][\"network\"] if rs else {}\n"),
])
print("ok")
