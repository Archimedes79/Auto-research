def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("tests/test_evaluation_api.py", [
    ('''        a = bev_eval.identity(run)
        assert a["checkpoint"] == "ckpt_00000010.pt" and a["size"] == 50_000 and a["adapter"] and a == bev_eval.identity(run), a
        data = bytearray(open(ck, "rb").read())
        data[10] ^= 1
        open(ck, "wb").write(bytes(data))
        assert bev_eval.identity(run)["fingerprint"] != a["fingerprint"], "a retrained checkpoint of the same name and size is another identity"
        open(os.path.join(run, "ckpt_00000020.pt"), "wb").write(bytes(data))
        assert bev_eval.identity(run)["checkpoint"] == "ckpt_00000020.pt", "the last checkpoint is the one evaluated"''', '''        a = bev_eval.identity(run)
        assert a["checkpoint"] == "ckpt_00000010.pt" and a["size"] == 50_000 and a["adapter"] and a == bev_eval.identity(run), a
        assert a["code"].startswith("live training/ "), "a run without a code snapshot is identified by the live training/ code it will be imported from"
        data = bytearray(open(ck, "rb").read())
        data[25_000] ^= 1                                                                                 # a byte in the MIDDLE of the file
        open(ck, "wb").write(bytes(data))
        assert bev_eval.identity(run)["sha1"] != a["sha1"], "a retrained checkpoint of the same name and size is another identity"
        open(os.path.join(run, "config.json"), "w").write('{"seed": 1}')
        assert bev_eval.identity(run)["config"] != a["config"], "another config is another identity"
        os.makedirs(os.path.join(run, "code"))
        open(os.path.join(run, "code", "SNAPSHOT.txt"), "w").write("commit abc123 (git archive; run x)")
        assert bev_eval.identity(run)["code"] == "commit abc123 (git archive; run x)", "a snapshot names the code"
        open(os.path.join(run, "ckpt_00000020.pt"), "wb").write(bytes(data))
        assert bev_eval.identity(run)["checkpoint"] == "ckpt_00000020.pt", "the last checkpoint is the one evaluated"'''),
    ('''    try:
        bev_eval.task("runs/none", ("maps", "mapz"))''', '''    from evalsuite.api import EvaluationError                                                              # one network per process: its code modules stay imported
    saved = dict(bev_eval._CODE)
    bev_eval._CODE["path"] = "/some/other/snapshot"
    try:
        bev_eval.source("runs/t16_full_s0", "val")
        raise AssertionError("a second code snapshot was accepted in one process")
    except EvaluationError as e:
        assert "one network per process" in str(e), e
    finally:
        bev_eval._CODE.update(saved)
    import torch                                                                                         # cuda asked for: moved when there, an error (never a quiet CPU run) when not
    from types import SimpleNamespace
    moved = []
    box = SimpleNamespace(to=lambda d: moved.append(d))
    fake = SimpleNamespace(_bev_subject=SimpleNamespace(device="cpu", model=box, loss=box))
    real_avail = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        try:
            bev_eval.subject("runs/x", fake, "cuda")
            raise AssertionError("cuda was not there and no error")
        except EvaluationError as e:
            assert "CUDA is not available" in str(e) and moved == [], e
        torch.cuda.is_available = lambda: True
        got = bev_eval.subject("runs/x", fake, "cuda")
        assert moved == ["cuda", "cuda"] and got.device == "cuda" and fake._bev_subject is None, moved
        try:
            bev_eval.subject("runs/x", fake, "cuda")
            raise AssertionError("a Subject was handed out twice")
        except RuntimeError as e:
            assert "handed out" in str(e)
    finally:
        torch.cuda.is_available = real_avail
    try:
        bev_eval.task("runs/none", ("maps", "mapz"))'''),
    ('''        c = r.card()
        assert c["split"] == "val"''', '''        os.chdir(os.path.join(ROOT, "..", "autoresearch"))                                                 # the documented working folder: the card must not depend on being in bev_fusion
        c = r.card()
        os.chdir(ROOT)
        snap = open(os.path.join(src_run, "code", "SNAPSHOT.txt")).read().split()
        assert c["commit"] == snap[snap.index("commit") + 1] and "seed" in c, (c.get("commit"), c.get("seed"))
        assert c["split"] == "val"'''),
    ('''        assert chk["ok"] and chk["frames_stored"] == 1 and chk["network_runs"] is False and chk["id"] == r["id"], chk''', '''        assert chk["ok"] and chk["frames_stored"] == 1 and chk["frames_to_run"] == 0 and chk["network_runs"] is False and chk["id"] == r["id"], chk'''),
])
print("ok")
