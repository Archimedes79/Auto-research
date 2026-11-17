p = "training/grid/test_jpast_train.py"
s = open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''    assert all(stream[i][1]["n"] == 0 and stream[i][1]["d2"] == 0.0 for i in (0, 4)) and all(stream[i][1]["n"] > 0 and stream[i][1]["d2"] > 0 and stream[i][1]["x2"] > 0 for i in (1, 2, 3, 5)), f"{label}: u sums"''',
    '''    assert all(stream[i][1]["n"] == 0 and stream[i][1]["d2_none"] == 0.0 for i in (0, 4)) and all(stream[i][1]["n"] > 0 and stream[i][1]["d2_none"] > 0 and stream[i][1]["x2"] > 0 for i in (1, 2, 3, 5)), f"{label}: u_none sums"
    assert all(stream[i][1]["n_other"] == 0 and stream[i][1]["d2_other"] == 0.0 for i in range(n)), f"{label}: a batch of one sample without a donor has no u"''')
rep('''    sh = {i: out for i, _, out, _ in E.iter_outputs(m, batches(), "shuffled", 3, donor=plan, states=states, frame_of=lambda j: frames[j])}''',
    '''    shi = {i: (out, info) for i, _, out, info in E.iter_outputs(m, batches(), "shuffled", 3, donor=plan, states=states, frame_of=lambda j: frames[j])}
    sh = {i: o_[0] for i, o_ in shi.items()}
    assert all(shi[i][1]["n_other"] == 0 for i in range(n)) and all(shi[i][1]["n"] > 0 for i in (1, 5)), f"{label}: in mode shuffled the donor is the past, no other-scene comparison"''')
rep('''    assert torch.equal(sh[0]["logp"], cold[0]) and torch.equal(sh[4]["logp"], cold[4]) and not torch.equal(sh[1]["logp"], stream[1][0]["logp"])''',
    '''    assert torch.equal(sh[0]["logp"], cold[0]) and torch.equal(sh[4]["logp"], cold[4]) and not torch.equal(sh[1]["logp"], stream[1][0]["logp"])
    # u in the stream read-out: the update with the stream's past against the update with the OTHER scene's past (the donor of the plan, warped by this frame's pose), by hand from the block's own inputs
    cap = {}
    hk = m.past_block.register_forward_hook(lambda mod, args, out: cap.__setitem__("args", args))
    try:
        for i, _, out, info in E.iter_outputs(m, batches(), "stream", 3, donor=plan, states=states, frame_of=lambda j: frames[j]):
            if not hp[i]:
                assert info["n_other"] == 0 and info["d2_other"] == 0.0 and info["x2_other"] == 0.0, f"{label}: frame {i} has no past"
                continue
            xin, pi, di, vi, pa = cap["args"]
            j = plan[i]
            pd_ = JP.make_past(dict(x=states[j][0], w=states[j][1]), frames[j], frames[i], R[i][None], t[i][None], True)
            with torch.no_grad():
                y_s, y_o, y_n = (m.past_block.delta(xin, pi, di, vi, q_) for q_ in (pa, pd_, None))
            mv = vi[..., None].float()
            want_o, want_n = float((((y_s - y_o) * mv) ** 2).sum()), float((((y_s - y_n) * mv) ** 2).sum())
            assert info["n_other"] == int(vi.sum()) == info["n"] and abs(info["d2_other"] - want_o) < 1e-4 * max(want_o, 1) and abs(info["d2_none"] - want_n) < 1e-4 * max(want_n, 1) and want_o > 0 and info["x2_other"] == info["x2"], f"{label}: u of frame {i}: {info} {want_o} {want_n}"
    finally:
        hk.remove()''')
rep('''    assert use["d2"][0] == 0.0 and use["d2"][1] > 0 and all(len(v) == 4 for v in use.values())''', '''    assert use["d2_none"][0] == 0.0 and use["d2_none"][1] > 0 and all(len(v) == 4 for v in use.values()) and set(use) >= set(E.JP.SUMS) | {"idx", "has_prev", "pos", "shift", "state_ss", "state_n"}''')
rep('''    mk = lambda keeps, links, d2, x2, pos, hp_: dict(meta=dict(run="r", split="val", idx=list(range(len(pos))), use=dict(d2=d2, x2=x2, pos=pos, has_prev=hp_, state_ss=[1.0] * len(pos), state_n=[1] * len(pos))), frames=[''', '''    mk = lambda keeps, links, d2, x2, pos, hp_, d2o, no_: dict(meta=dict(run="r", split="val", idx=list(range(len(pos))), use=dict(d2_none=d2, x2=x2, n=[int(h) for h in hp_], d2_other=d2o, x2_other=[x if k else 0.0 for x, k in zip(x2, no_)], n_other=no_, pos=pos, has_prev=hp_,
                                                                                                                                                           state_ss=[1.0] * len(pos), state_n=[1] * len(pos))), frames=[''')
rep('''    s_ = mk([[1, 1, 1], [1, 0, 0]], [[(0, 1), (1, 2)], [(0, 1)]], [0.0, 1.0], [4.0, 4.0], [0, 1], [False, True])
    o_ = mk([[1, 0, 1], [1, 0, 0]], [[(0, 1)], [(0, 1)]], [0.0, 0.0], [4.0, 4.0], [0, 0], [False, False])
    rep = E.use_report(s_, o_)
    assert abs(rep["u"] - 0.5) < 1e-9 and rep["frames_with_past"] == 1 and abs(rep["u_by_position"]["1-1"] - 0.5) < 1e-9 and np.isnan(rep["u_by_position"]["2-2"]) and rep["against_off"] == dict(kept_gained_per_frame=0.5, kept_lost_per_frame=0.0, kept_flips_per_frame=0.5,
                                                                                                                                                                 frames_with_a_flip=0.5, links_changed_per_frame=0.5), rep''', '''    s_ = mk([[1, 1, 1], [1, 0, 0], [1, 0, 0]], [[(0, 1), (1, 2)], [(0, 1)], [(0, 1)]], [0.0, 1.0, 6.0], [4.0, 4.0, 16.0], [0, 1, 2], [False, True, True], [0.0, 9.0, 0.0], [0, 2, 0])      # frame 2 has a past but no other scene's past
    o_ = mk([[1, 0, 1], [1, 0, 0], [1, 0, 0]], [[(0, 1)], [(0, 1)], [(0, 1)]], [0.0, 0.0, 0.0], [4.0, 4.0, 4.0], [0, 0, 0], [False, False, False], [0.0] * 3, [0] * 3)
    rep = E.use_report(s_, o_)
    assert abs(rep["u"] - 1.5) < 1e-9 and abs(rep["u_none"] - math.sqrt(7 / 20)) < 1e-9 and rep["frames_with_past"] == 2 and rep["frames_with_other_past"] == 1, rep
    assert abs(rep["u_by_position"]["1-1"] - 1.5) < 1e-9 and np.isnan(rep["u_by_position"]["2-2"]) and abs(rep["u_none_by_position"]["1-1"] - 0.5) < 1e-9 and abs(rep["u_none_by_position"]["2-2"] - math.sqrt(6 / 16)) < 1e-9 and np.isnan(rep["u_none_by_position"]["3-3"]), rep
    assert rep["against_off"] == dict(kept_gained_per_frame=1 / 3, kept_lost_per_frame=0.0, kept_flips_per_frame=1 / 3, frames_with_a_flip=1 / 3, links_changed_per_frame=1 / 3), rep["against_off"]''')
rep('''use=dict(d2=[0.0, 1.0, 4.0, 1.0, 1.0], x2=[4.0] * 5, pos=[0, 1, 5, 4, 1], has_prev=[False, True, True, True, True], state_ss=[8.0, 18.0, 50.0, 18.0, 36.0], state_n=[2] * 5)), frames=[])''',
    '''use=dict(d2_none=[0.0, 1.0, 4.0, 1.0, 1.0], x2=[4.0] * 5, n=[0, 1, 1, 1, 1], d2_other=[0.0] * 5, x2_other=[0.0] * 5, n_other=[0] * 5, pos=[0, 1, 5, 4, 1], has_prev=[False, True, True, True, True], state_ss=[8.0, 18.0, 50.0, 18.0, 36.0], state_n=[2] * 5)), frames=[])''')
rep('''        assert rep["frames_with_past"] == 11 and rep["u"] > 0 and rep["against_off"]["kept_flips_per_frame"] >= 0''', '''        assert rep["frames_with_past"] == 11 and rep["frames_with_other_past"] == 11 and rep["u"] > 0 and rep["u_none"] > 0 and rep["u"] != rep["u_none"] and rep["against_off"]["kept_flips_per_frame"] >= 0''')
rep('''"u by position counts the cold start", "u_by_position={f\\"{lo}-{hi}\\" if hi < 10 ** 9 else f\\"{lo}+\\": ratio((pos >= lo) & (pos <= hi) & (hp > 0))", "u_by_position={f\\"{lo}-{hi}\\" if hi < 10 ** 9 else f\\"{lo}+\\": ratio((pos >= lo) & (pos < hi))"''', '''"the u buckets exclude their upper bound", "bucket = {f\\"{lo}-{hi}\\" if hi < 10 ** 9 else f\\"{lo}+\\": (pos >= lo) & (pos <= hi) & (hp > 0)", "bucket = {f\\"{lo}-{hi}\\" if hi < 10 ** 9 else f\\"{lo}+\\": (pos >= lo) & (pos < hi) & (hp > 0)"''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
