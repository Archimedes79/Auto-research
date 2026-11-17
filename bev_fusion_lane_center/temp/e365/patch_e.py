import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


# (9) three branches
rep('''    print("decode_fork: record keeps both branches''', '''    out, tok = stub_out(K, {0: {1: 0.25, 2: 0.25, 3: 0.25}}, [0.1] * K)                              # three weak branches and none, all 0.25: nothing reaches 0.3
    _, f = pred_of(out, tok)
    assert len(f["rec"]["src"]) == 0, f["rec"]
    assert list(zip(f["argmax"]["src"].tolist(), f["argmax"]["dst"].tolist())) == [(0, 1)], "a tie goes to the first key (torch argmax), one link: the argmax cap"
    print("decode_fork: record keeps both branches''')

# (7) mutants
rep('''    def with_assign(frame, keep, e_node, links, flags=None):''', '''    gt_links = dict(src=np.array([1]), dst=np.array([0]), p=np.ones(1), long=np.zeros(1, bool), a=np.full(1, 1 / 3), b=np.full(1, 1 / 3), curves=None)
    for name, leak in (("links replaced by GT links", lambda p_: p_.update(links=gt_links)), ("e set to 1", lambda p_: p_["nodes"].update(e=np.ones_like(p_["nodes"]["e"]))),
                       ("cell changed", lambda p_: p_["nodes"].update(cell=p_["nodes"]["cell"] + 1))):
        def mutant2(frame, keep, e_node, links, flags=None, leak=leak):
            p_ = real(frame, keep, e_node, links, flags)
            leak(p_)
            return p_
        with patched((LP, "to_pred", mutant2)):
            must_fail(E.frame_prediction, tok, f, "rec")
    def with_assign(frame, keep, e_node, links, flags=None):''')
rep("is caught by check_writer (6 mutants)", "is caught by check_writer (6 mutants), links replaced by GT links / e set to 1 / a changed cell by frame_prediction (3 more)")

# numbers: add checks
rep('''    print("numbers: layer-1 comparison at 1e-6''', '''    # parity: the valid tokens only one side has
    torch.manual_seed(0)
    m = JK.JTokenHead().eval()
    m.set_step(JK.SWITCH)
    sb = JK.synthetic_batch(2, 48, seed=3)
    cells = np.tile(np.arange(48), (2, 1))
    live = {k: sb[k].clone() for k in JK.INPUTS}
    live["cell"] = torch.from_numpy(cells.copy())
    ok, num = E.compare_sets(m, sb, cells, live, 10.0)
    assert ok and num["unmatched_share"] == 0.0 and num["max_dp"] == 0.0, num
    live["tok"] = live["tok"] + 1e-3                                                                 # a small difference in the inputs: probabilities agree within the tolerance
    assert E.compare_sets(m, sb, cells, live, 0.05)[0]
    live["cell"][:, :5] = 1000 + torch.arange(5)                                                      # 5 tokens per frame are other cells: they are valid on one side only
    ok, num = E.compare_sets(m, sb, cells, live, 10.0)
    assert not ok and num["unmatched_share"] > 10 * E.UNMATCHED_LIMIT and num["valid_cache_not_live"] > 0 and num["valid_live_not_cache"] > 0, num
    # gpu_free: this process and its parents never count as another run.py job
    fake = lambda out: types.SimpleNamespace(stdout=out)  # noqa: E731
    mine = "\\n".join(str(x) for x in (os.getpid(), os.getppid()))
    with patched((E.subprocess, "run", lambda cmd, **k: fake("500\\n") if cmd[0] == "nvidia-smi" else fake(mine + "\\n"))):
        assert E.gpu_free()[0], E.gpu_free()
    with patched((E.subprocess, "run", lambda cmd, **k: fake("500\\n") if cmd[0] == "nvidia-smi" else fake(mine + "\\n999999\\n"))):
        ok, why = E.gpu_free()
        assert not ok and "999999" in why and str(os.getpid()) not in why, why
    with patched((E.subprocess, "run", lambda cmd, **k: fake("2500\\n") if cmd[0] == "nvidia-smi" else fake(""))):
        assert not E.gpu_free()[0]
    # stamped result files, the dump identity, the control card names
    with tempfile.TemporaryDirectory() as d:
        a1, a2 = E._write_stamped(d, "x", dict(v=np.float32(1.5))), E._write_stamped(d, "x", dict(v=2))
        assert a1 != a2 and json.load(open(a1)) == dict(v=1.5) and len(os.listdir(d)) == 2, "a result file is never skipped or overwritten"
        open(os.path.join(d, "f.bin"), "wb").write(b"abc")
        fid = E._file_id(os.path.join(d, "f.bin"))
        assert fid["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and fid["mtime"].endswith("UTC"), fid
        assert E.ctl_name("hi", "e292_a", 0, 384) == "hi_e292_a_s0_K384.json" and E.ctl_name("hi", "e292_a", 0, 384, 3) == "hi_e292_a_s0_K384_n3.json"
        card = lambda K: dict(metrics=dict(topo_F1=0.8), info=dict(K=K))  # noqa: E731
        json.dump(card(384), open(os.path.join(d, "hi_e292_a_s0_K384.json"), "w"))
        json.dump(card(384), open(os.path.join(d, "rule_e292_a_s0.json"), "w"))                      # an earlier card without K in its name counts only when its K is this one
        assert E.read_ctl("hi", "e292_a", 0, 384, d) == 0.8 and E.read_ctl("hi", "e292_a", 0, 512, d) is None and E.read_ctl("rule", "e292_a", 0, 384, d) == 0.8 and E.read_ctl("rule", "e292_a", 0, 512, d) is None
    # layer 1 stores plain B's records only when it passes; score refuses without them
    with tempfile.TemporaryDirectory() as d:
        base = {k: 0.5 for k in E.KEYS}
        json.dump(dict(variants=dict(primary=dict(numbers=dict(base=base)))), open(os.path.join(d, "card_s{seed}.json".format(seed=9)), "w"))
        res = lambda off: dict(graph={k: v + off for k, v in base.items()}, dump=dict(path="x", sha256="0", mtime="t"))  # noqa: E731
        with patched((E, "BASE_CARD", os.path.join(d, "card_s{seed}.json")), (E, "SCRATCH", d), (E, "plain_b", lambda seed, dump=None: res(0.0))):
            assert E.check_layer1(types.SimpleNamespace(seed=9, dump=None)) == 0 and os.path.exists(E.planb_path(9)) and E.stored_plain_b(9)["dump"]["sha256"] == "0"
            os.remove(E.planb_path(9))
            with patched((E, "plain_b", lambda seed, dump=None: res(1e-3))):
                assert E.check_layer1(types.SimpleNamespace(seed=9, dump=None)) == 1 and not os.path.exists(E.planb_path(9)), "a FAIL stores nothing"
            must_fail(E.stored_plain_b, 9)
    print("numbers: layer-1 comparison at 1e-6''')
open(p, "w", encoding="utf-8").write(s)
