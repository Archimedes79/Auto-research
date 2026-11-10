def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("../bev_fusion/tests/test_evaluation_api.py", [
    ('''    assert hasattr(bev_eval, "reference") and hasattr(bev_eval, "card") and hasattr(bev_eval, "time") and hasattr(bev_eval, "probe")''', '''    assert hasattr(bev_eval, "reference") and hasattr(bev_eval, "card") and hasattr(bev_eval, "time") and hasattr(bev_eval, "probe")
    assert hasattr(bev_eval, "free_source") and hasattr(bev_eval, "free_subject"), "the Runner's pair: both or none"'''),
    ('''    # the identity of a network: its checkpoint (name, size, fingerprint), config, code and the adapter; a retrained checkpoint of the same name is another identity''', '''    # the identity of a network: its checkpoint (name, fingerprint), config, code and the adapter (bev_adapter.py and bev_free.py); a retrained checkpoint of the same name is another identity'''),
    ('''        assert a["checkpoint"] == "ckpt_00000010.pt" and a["size"] == 50_000 and a["adapter"] and a == bev_eval.identity(run), a''', '''        assert a["checkpoint"] == "ckpt_00000010.pt" and a["adapter"] and a == bev_eval.identity(run), a
        file_hash = bev_eval._file_hash
        bev_eval._file_hash = lambda p: file_hash(p) + ("x" if p.endswith("bev_free.py") else "")           # an edit of bev_free.py (the Runner's path) is another identity
        try:
            assert bev_eval.identity(run)["adapter"] != a["adapter"], "bev_free.py is part of the identity"
        finally:
            bev_eval._file_hash = file_hash'''),
])
print("ok")
