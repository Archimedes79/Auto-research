p = "training/grid/test_jtoken_fast.py"
t = open(p, encoding="utf-8").read()

# 1. test_trainer: CUDA hidden inside the test, FT.main with its fast_path.json, a plain checkpoint resumed by the fast trainer
a = t.index("def test_trainer():")
b = t.index("def test_gpu():")
new_trainer = '''def test_trainer():
    """20 smoke steps (synthetic batches, K 48, bs 4, CPU fp32: CUDA is hidden inside the test) of the plain trainer and of `jtoken_fast_train.main` (the same run arguments): the final weights agree to 2e-3 (3e-6 in practice),
    the fast run folder has fast_path.json, it is written only after the recipe check (a folder of another recipe is refused and gets none), a plain run's finished checkpoint is resumed by the fast trainer; the patches are undone"""
    import tempfile
    sys.path.insert(0, os.path.join(HERE, "..", "train"))
    import jtoken_train as JT
    import jtoken_fast_train as FT
    import trainer as TR
    keep = (JT.build_head, JK.jtoken_loss, JT.check_start, JT.train, TR.RUNS, torch.cuda.is_available)
    torch.cuda.is_available = lambda: False
    try:
        with tempfile.TemporaryDirectory() as d:
            argv = ["--smoke", "--bs", "4", "--none-weight", "0.25"]
            JT.train(JT.parse_args(["--name", "plain"] + argv), d, 500, 48)
            orig_train = JT.train
            JT.train = lambda a, runs, *x, **k: orig_train(a, runs, 500, 48)                   # the smoke run with K 48 through `main`
            TR.RUNS = d
            FT.main(["--name", "fast"] + argv)
            sa, sb = (torch.load(os.path.join(d, n, "model.pt"), map_location="cpu", weights_only=False) for n in ("plain", "fast"))
            assert sa["state"].keys() == sb["state"].keys()
            worst = max(float((sa["state"][k].float() - sb["state"][k].float()).abs().max()) for k in sa["state"])
            assert worst < 2e-3, f"after 20 steps the weights differ by {worst}"
            note = json.load(open(os.path.join(d, "fast", "fast_path.json")))
            assert note["head"] == "jtoken_fast.FastJTokenHead" and "CPU" in note["loader"], note
            os.makedirs(os.path.join(d, "other"))                                                # a folder that holds ANOTHER recipe: refused by the start check, no fast_path.json written into it
            shutil.copy(os.path.join(d, "plain", "config.json"), os.path.join(d, "other", "config.json"))
            try:
                FT.main(["--name", "other", "--smoke", "--bs", "4", "--none-weight", "0.5"])
                raise SystemExit("the start check must refuse another recipe")
            except AssertionError:
                pass
            assert not os.path.exists(os.path.join(d, "other", "fast_path.json"))
            shutil.copytree(os.path.join(d, "plain"), os.path.join(d, "resume"))                 # a finished plain run resumed by the fast trainer (the same recipe: the name is not in it)
            FT.main(["--name", "resume"] + argv)
            assert "resumed from" in open(os.path.join(d, "resume", "log.txt")).read(), "the fast trainer resumes a plain checkpoint"
    finally:
        JT.build_head, JK.jtoken_loss, JT.check_start, JT.train, TR.RUNS, torch.cuda.is_available = keep
    assert JT.build_head is keep[0] and JK.jtoken_loss is keep[1] and JT.check_start is keep[2]
    print(f"trainer ok (max weight difference after 20 steps {worst:.2e})")


'''
t = t[:a] + new_trainer + t[b:]
if "import shutil" not in t.split("def ")[0]:
    t = t.replace("import os\nimport sys\n", "import json\nimport os\nimport shutil\nimport sys\n", 1)

# 3. M1: a scale pair outside the long gate must not count
a = '''    _close(l1, l2, 1e-5, "sparse loss")
    assert "scale_l1" in i2 and float(i2["scale_l1"]) > 0
'''
assert t.count(a) == 1
t = t.replace(a, a + '''    out_gate = dict(bs, sp_b=torch.cat([bs["sp_b"], torch.tensor([0])]), sp_i=torch.cat([bs["sp_i"], torch.tensor([0])]), sp_j=torch.cat([bs["sp_j"], torch.tensor([0])]),     # the pair (0, 0): never in a gate
                    sp_y=torch.cat([bs["sp_y"], torch.tensor([[0.9, 0.9]])]), sp_ok=torch.cat([bs["sp_ok"], torch.tensor([True])]))
    assert not bool(fast(out_gate)["cand"][0, 1, 0, 0])
    l2b, _ = JF.fast_loss(fast(out_gate), out_gate)
    _close(l2, l2b, 1e-6, "a pair outside the long gate must not count (the cand mask of the scale term)")
''')

# 4. teachers: the eval-mode dense scale switched off
a = '''def test_delegation():'''
assert t.count(a) == 1
t = t.replace(a, '''def test_scale_in_eval():
    plain, fast, b = _pair(9)
    fast.eval()
    with torch.no_grad():
        assert fast(b)["scale"] is not None
        fast.scale_in_eval = False
        o = fast(b)
    assert o["scale"] is None and "scale_at" in o
    fast.train()
    l1, _ = JK.jtoken_loss(plain(b), b)
    l2, _ = JF.fast_loss(fast(b), b)
    _close(l1, l2, 1e-5, "the loss does not read the dense scale")
    print("scale_in_eval ok")


''' + a)
t = t.replace("test_sparse, test_frame_mean, test_delegation,", "test_sparse, test_frame_mean, test_scale_in_eval, test_delegation,")
open(p, "w", encoding="utf-8").write(t)
print("ok")
