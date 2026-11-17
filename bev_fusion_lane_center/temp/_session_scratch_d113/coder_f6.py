p = "training/grid/test_jpast_train.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
# state of the stream frames + boundary
rep('''    trunc = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), "truncated", 3)}''', '''    for i in (0, 2):                                                                                         # the CARRIED state: the final token states x of the frame's valid tokens (not the block's input)
        o_i, inf = stream[i]
        v_i = o_i["valid"][0].bool()
        assert inf["state_n"] == int(v_i.sum()) * 64 and inf["state_ss"] > 0 and abs(inf["state_ss"] - float((o_i["x"][0] ** 2)[v_i].sum())) < 1e-4 * inf["state_ss"] and abs(inf["state_ss"] - inf["x2"]) > 1e-3 * inf["state_ss"], f"{label}: the carried state's sums"
    for s0, learned in ((JK.SWITCH - 1, False), (JK.SWITCH, True)):                                          # the read-out's switch of the key weight at the boundary (as the head's and make_past's)
        m.set_step(s0)
        calls_b, restore = spy(m)
        list(E.iter_outputs(m, batches()[:2], "stream", 3))
        restore()
        assert torch.equal(calls_b[1]["past"]["w"], calls_b[0]["out"]["w"] if learned else frames[0]["e"].sigmoid()), f"{label}: the read-out's key weight at counter {s0}"
    m.set_step(JK.SWITCH + 10)
    trunc = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), "truncated", 3)}''')
rep('''use=dict(d2=d2, x2=x2, pos=pos, has_prev=hp_))''', '''idx=list(range(len(pos))), use=dict(d2=d2, x2=x2, pos=pos, has_prev=hp_, state_ss=[1.0] * len(pos), state_n=[1] * len(pos)))''')
rep('''    assert E.is_cold("stream", True, 5, 3) is False''', '''    m2 = dict(meta=dict(run="r", split="val", idx=[10, 11, 12, 13], use=dict(d2=[0.0, 1.0, 4.0, 1.0], x2=[4.0] * 4, pos=[0, 1, 5, 1], has_prev=[False, True, True, True], state_ss=[8.0, 18.0, 50.0, 18.0], state_n=[2, 2, 2, 2])), frames=[])
    r2 = E.use_report(m2)                                                                                    # the carried state's RMS per stream position and the first-5 split, by hand
    sr = r2["state_rms_by_position"]
    assert len(sr) == 6 and sr[0] == 2.0 and sr[1] == 3.0 and sr[5] == 5.0 and all(np.isnan(sr[q]) for q in (2, 3, 4)), sr
    assert r2["position_per_frame"] == [0, 1, 5, 1] and r2["frame_idx_first5"] == [10, 11, 13] and r2["frame_idx_later"] == [12], r2
    assert E.is_cold("stream", True, 5, 3) is False''')
# dump on real frames: sibling folders
rep('''        paths = {}
        for mode in ("stream", "off", "truncated"):
            paths[mode] = E.dump(argparse.Namespace(run="x", seed=0, split="val", mode=mode, window=3, cpu=True, max=12), runs=tmp)
        assert [os.path.basename(p) for p in paths.values()] == ["token_val_n12.pkl", "token_val_off_n12.pkl", "token_val_truncated_n12.pkl"]
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=0, split="val", mode="stream", window=3, cpu=True, max=12), runs=tmp), "never overwritten")''', '''        paths = {}
        for mode in ("stream", "off", "truncated"):
            paths[mode] = E.dump(argparse.Namespace(run="x", seed=0, split="val", mode=mode, window=3, cpu=True, max=12), runs=tmp)
        assert [os.path.relpath(p, tmp) for p in paths.values()] == ["x/token_val_n12.pkl", "x_off/token_val_n12.pkl", "x_truncated/token_val_n12.pkl"], paths
        assert all(open(os.path.join(tmp, f"x_{md}", "model.pt"), "rb").read() == open(os.path.join(tmp, "x", "model.pt"), "rb").read() for md in ("off", "truncated")), "the sibling folder holds a copy of model.pt"
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=0, split="val", mode="stream", window=3, cpu=True, max=12), runs=tmp), "never overwritten")
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=1, split="val", mode="stream", window=3, cpu=True, max=5), runs=tmp), "does not match the run")''')
rep('''        assert {"run", "seed", "split", "drop", "drop_seed", "idx", "every", "tokens", "K", "ident", "cache", "steps"} <= set(meta) and meta["idx"] == list(range(12)) and meta["mode"] == "stream" and meta["every"] == 1 and meta["drop"] == 0.0''', '''        assert {"run", "seed", "split", "drop", "drop_seed", "idx", "every", "tokens", "K", "ident", "cache", "steps", "model"} <= set(meta) and meta["idx"] == list(range(12)) and meta["mode"] == "stream" and meta["every"] == 1 and meta["drop"] == 0.0 and meta["run"] == "x" and meta["model"] == "model.pt"
        assert d["off"]["meta"]["run"] == "x_off" and d["off"]["meta"]["of_run"] == "x" and d["truncated"]["meta"]["run"] == "x_truncated", "the meta names the sibling run the scorer reads"
        for sib in ("x", "x_off"):                                                                          # the scorer's own reader accepts the dumps: the cache, the kept nodes, the scorer's frames (a mode dump is read as a run of its own)
            r_ = E.JE.read_dump(argparse.Namespace(run=sib, seed=0, split="val", model="model.pt", every=1, drop=0.0, drop_seed=0), tmp, 12)
            assert len(r_["fr"]) == 12 and r_["idx"].tolist() == list(range(12)), sib''')
rep('''        rep = E.use_report(d["stream"], d["off"])
        assert rep["frames_with_past"] == 11 and rep["u"] > 0 and rep["against_off"]["kept_flips_per_frame"] >= 0''', '''        rep = E.use_report(d["stream"], d["off"])
        assert rep["frames_with_past"] == 11 and rep["u"] > 0 and rep["against_off"]["kept_flips_per_frame"] >= 0
        res = E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp)                              # the command: stream + the sibling x_off -> past_use_val_n12.json
        assert os.path.exists(os.path.join(tmp, "x", "past_use_val_n12.json")) and res["position_per_frame"] == list(range(12)) and res["frame_idx_first5"] == [0, 1, 2, 3, 4] and res["frame_idx_later"] == list(range(5, 12)) and "against_off" in res
        assert len(res["state_rms_by_position"]) == 12 and all(v > 0 for v in res["state_rms_by_position"]), res["state_rms_by_position"]
        refuses(lambda: E.use(argparse.Namespace(run="x", split="val", max=12), runs=tmp), "never overwritten")
        with open(os.path.join(tmp, "x_off", "model.pt"), "ab") as fh:                                       # a sibling folder holds the same model.pt as the run or the dump is refused
            fh.write(b"0")
        refuses(lambda: E.dump(argparse.Namespace(run="x", seed=0, split="val", mode="off", window=3, cpu=True, max=5), runs=tmp), "is not a copy")''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
