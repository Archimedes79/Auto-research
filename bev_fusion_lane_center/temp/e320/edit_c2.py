def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, old[:70], s.count(old))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)

block = '''    # E320, critic 2026-11-12 (three mutations survived the first self-test): the candidate gate, the number of candidates, the heading mix and the direct position gradient, on a scene built for them
    cg = -50.0 + (torch.arange(50).float() + 0.5) * 2.0
    n2 = dict(e=torch.full((1, 50, 50), -8.0), flags=torch.full((1, 8, 50, 50), -8.0), uout=torch.full((1, 50, 50), -8.0), uin=torch.full((1, 50, 50), -8.0), s=torch.full((1, 50, 50), -8.0),
              pos=torch.stack([cg.view(50, 1).expand(50, 50), cg.view(1, 50).expand(50, 50)], 0)[None].clone(), dir_raw=torch.zeros(1, 2, 50, 50), tok=torch.randn(1, 64, 50, 50))
    n2["dir_raw"][:, 1] = 1.0
    for i_ in (23, 24, 25):                                                                                   # a block of 12 existing nodes around the keypoint, all heading (0, 1)
        for j_ in (19, 20, 21, 22):
            n2["e"][:, i_, j_] = 4.0
    rot = lambda deg: (-math.sin(math.radians(deg)), math.cos(math.radians(deg)))  # noqa: E731
    n2["dir_raw"][0, :, 23, 21] = torch.tensor(rot(60.0))                                                   # 60 degrees off: outside the 45 degree gate
    n2["dir_raw"][0, :, 25, 21] = torch.tensor(rot(20.0))                                                   # 20 degrees off: inside the gate, it must turn the attached heading
    kpA, kpB = torch.tensor([-1.8, -7.9]), torch.tensor([30.0, 30.0])                                          # A has more than ATT_M gated nodes, B has none
    pos2 = torch.stack([kpA, kpB])[None].clone().requires_grad_(True)
    D2 = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    at = Attach(d=64)
    for p_ in at.parameters():
        nn.init.zeros_(p_)
    with torch.no_grad():
        at.dust.bias.fill_(-20.0)                                                                             # no dustbin: the weights are uniform over the valid candidates (zero scores)
    o2 = at(torch.zeros(1, 2, 64), pos2, D2, torch.zeros(1, 2), n2)
    npos2, ndir2 = n2["pos"][0].flatten(1).t(), torch.nn.functional.normalize(n2["dir_raw"][0].flatten(1).t(), dim=-1)
    dd2 = (npos2 - kpA).norm(dim=-1)
    gate2 = (dd2 <= ATT_REACH) & ((ndir2 * D2[0, 0]).sum(-1) >= math.cos(math.radians(ATT_ANG))) & (n2["e"][0].flatten() > -2.9)
    assert int(gate2.sum()) > ATT_M, int(gate2.sum())                                                         # more gated nodes than the cap: the nearest ATT_M must be kept
    want = npos2[gate2][dd2[gate2].argsort()[:ATT_M]]
    got = o2["att_cpos"][0, 0][o2["att_cvalid"][0, 0]]
    assert o2["att_cvalid"][0, 0].sum() == ATT_M and torch.allclose(got, want, atol=1e-5), (got, want)
    assert not any(torch.allclose(row, npos2[23 * 50 + 21]) for row in got), "the node turned by 60 degrees must be gated out"
    assert not o2["att_cvalid"][0, 1].any() and torch.allclose(o2["att_pos"][0, 1], kpB), "a keypoint with no node in reach keeps its position"
    mean_dir = torch.nn.functional.normalize(ndir2[gate2][dd2[gate2].argsort()[:ATT_M]].mean(0), dim=0)
    dev_deg = math.degrees(math.acos(float(torch.clamp((o2["att_dir"][0, 0] * D2[0, 0]).sum(), -1, 1))))
    assert float((o2["att_dir"][0, 0] - mean_dir).abs().max()) < 1e-4 and dev_deg > 1.0, (o2["att_dir"][0, 0], mean_dir, dev_deg)    # the 20 degree node turns the attached heading: ignoring the heading mix gives dev_deg = 0
    for p_ in at.parameters():                                                                               # the direct gradient of the attached position w.r.t. the keypoint: w0 per coordinate (the scores are constants here)
        nn.init.zeros_(p_)
    o3 = at(torch.zeros(1, 2, 64), pos2, D2, torch.zeros(1, 2), n2)
    o3["att_pos"].sum().backward()
    w0A = 1.0 / (ATT_M + 1.0)
    assert abs(float(pos2.grad[0, 0].sum()) - 2 * w0A) < 1e-5 and abs(float(pos2.grad[0, 1].sum()) - 2.0) < 1e-5, pos2.grad                  # w0 x keypoint carries a gradient (cut it with detach and this is 0)
'''
edit("temp/e320/dev/jends.pysrc", [
    ('''    print("jends selftest ok; params"''', block + '''    print("jends selftest ok; params"'''),
])
edit("temp/e320/dev/jends_train.pysrc", [
    ('''(f", att_acc {info['att_acc']:.2f}, att_ce {info['att_ce']:.2f}" if "att_ce" in info else "")''', '''(f", attachment over {info['att_n']:.0f} keypoints with a candidate: att_acc {info['att_acc']:.2f} (nearest-candidate baseline {info['att_acc_nearest']:.2f}), att_ce {info['att_ce']:.2f}, attached end error {info['att_d']:.2f} m against the keypoint's {info['att_kp_d']:.2f} m (train frames, GT-injected keypoints included)" if "att_ce" in info else "")'''),
])
print("c2 ok")
