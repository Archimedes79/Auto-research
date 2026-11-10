def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:120])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evalsuite/selftest_api.py", [
    ('''    assert os.path.isfile(ev.report(allr["id"])) and ev.make_card(allr["id"])["name"] == allr["id"]''', '''    assert os.path.isfile(ev.report(allr["id"]))
    # the card is a FILE a decision can read: default next to the network's eval folder, or where `out` says (a file, or a folder), atomic; the dict is returned with its path
    m = ev.make_card(allr["id"])
    eid = allr["id"].partition("@")[2]
    assert m["card"]["name"] == allr["id"] and m["path"] == os.path.join(out(net), f"card_{eid}.json") and json.load(open(m["path"])) == json.loads(json.dumps(m["card"], default=str))
    to_file = ev.make_card(allr["id"], out=os.path.join(STATE.root, "cards", "mine.json"))
    to_dir = ev.make_card(allr["id"], out=os.path.join(STATE.root, "cards"))
    assert os.path.isfile(to_file["path"]) and to_dir["path"] == os.path.join(STATE.root, "cards", f"card_{eid}.json") and os.path.isfile(to_dir["path"])
    assert not [f for f in os.listdir(os.path.join(STATE.root, "cards")) if ".tmp" in f], "atomic: no half-written file stays"'''),
    # ---- an evaluator that cannot be hashed fails closed
    ('''    # the evaluation of an id that is being written is refused with the holder's name''', '''    # an adapter that gives no evaluator hash (a file that shapes the numbers is missing or renamed) fails closed: nothing is reused across metric edits that nobody can see
    STATE.ev = None
    try:
        _raises(EvaluationError, "no evaluator hash", ev.run, net, "val", progress=0)
        chk_ev = ev.check(net, "val")
        assert not chk_ev["ok"] and any("no evaluator hash" in p for p in chk_ev["problems"]), chk_ev
        assert any("cannot be hashed now" in w for w in ev.show_numbers(allr["id"])["warnings"]), "an evaluation made earlier says its code can no longer be compared"
    finally:
        STATE.ev = "x"
    # the evaluation of an id that is being written is refused with the holder's name'''),
])
patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''    assert len(bev_eval.evaluator_extra()) == 12, "the adapter's own code is part of the evaluation id"''', '''    assert len(bev_eval.evaluator_extra()) == 12, "the adapter's own code is part of the evaluation id"
    from evalsuite.api import EvaluationError
    from evaluation import bev_adapter                                                                   # an evaluator that cannot be hashed fails closed (a renamed file of NUMBER_PATHS)
    saved_paths = bev_adapter.NUMBER_PATHS
    bev_adapter.NUMBER_PATHS = saved_paths + (os.path.join(ROOT, "evaluation", "no_such_file_renamed.py"),)
    try:
        assert bev_eval.evaluator_now() is None
        try:
            bev_eval.card({"id": "x@e1", "evaluator": "abc123", "results": "x@r1"}, None, None, None)
            raise AssertionError("a card was made without a comparable evaluator hash")
        except EvaluationError as e:
            assert "other evaluator code" in str(e) and "None" in str(e), e
    finally:
        bev_adapter.NUMBER_PATHS = saved_paths''' ),
    ('''        file_hash = bev_eval._file_hash
        bev_eval._file_hash = lambda p: file_hash(p) + ("x" if p.endswith("bev_free.py") else "")           # an edit of bev_free.py (the Runner's path) is another identity
        try:
            assert bev_eval.identity(run)["adapter"] != a["adapter"], "bev_free.py is part of the identity"
        finally:
            bev_eval._file_hash = file_hash''', '''        file_hash = bev_eval._file_hash
        for edited in ("bev_free.py", "runner.py", "torch_taps.py"):                                       # an edit of the label-free path or of the frame loop / taps is another identity
            bev_eval._file_hash = lambda p, edited=edited: file_hash(p) + ("x" if p.endswith(edited) else "")
            try:
                assert bev_eval.identity(run)["adapter"] != a["adapter"], f"{edited} is part of the identity"
            finally:
                bev_eval._file_hash = file_hash'''),
    ('''        c = ev.make_card(r["id"])
''', '''        made = ev.make_card(r["id"])
        c = made["card"]
        chk_card = subprocess.run([sys.executable, os.path.join(ROOT, "..", "autoresearch", "tools", "card.py"), "check", made["path"]], capture_output=True, text=True, timeout=120)
        assert chk_card.returncode == 0, (made["path"], chk_card.stdout[-300:], chk_card.stderr[-300:])      # the written file is a valid card of the contract
'''),
])
print("ok")
