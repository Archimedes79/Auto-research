"""E339 revision after the critic (C1, C2, point 4): soft-target BCE instead of CWD (a spatial softmax drops the weak ends of a sparse end map), the heading term only where the teacher sees an end,
a selftest that tests the detach, the mask and a real fall, an assert that the teacher sits on the same pyramid."""


def rd(p):
    return open(p, encoding="utf-8").read()


def wr(p, s):
    open(p, "w", encoding="utf-8", newline="\n").write(s)


def sub(s, a, b):
    assert s.count(a) == 1, a[:80]
    return s.replace(a, b)


p = "training/grid/jmaps.py"
s = rd(p)
i0 = s.index("def cwd(s, t, tau=1.0):")
i1 = s.index("@torch.no_grad()\ndef keypoints(")
s = s[:i0] + '''def distill_loss(out, tout, pos_w=20.0, peak=0.3):
    """E339: a narrower student follows the frozen teacher's maps (the same pyramid features): the entry / exit heat logits (channels 2:4) against the teacher's heat PROBABILITIES as soft targets, BCE with the positive weight of the ground-truth
    heat loss (a softmax over the cells would give a weak end next to a strong one a fraction of a percent of the mass; here every cell keeps its weight), + the cosine of the student's heading to the teacher's heading at the cells where the teacher
    sees an end (heat probability > peak; the headings are trained at the ground-truth ends only, elsewhere they are noise), weight 0.5 as in `jmaps_loss`; zone / core / paint keep their ground-truth losses only -> (loss, info)"""
    s, t = out["j1"].float(), tout["j1"].detach().float()
    pt = t[:, 2:4].sigmoid()
    l_h = F.binary_cross_entropy_with_logits(s[:, 2:4], pt, pos_weight=torch.tensor(pos_w, device=s.device))
    l_d = 0.0
    for k in range(2):
        ps, th = s[:, 4 + 2 * k: 6 + 2 * k], t[:, 4 + 2 * k: 6 + 2 * k]
        ps = ps / ps.norm(dim=1, keepdim=True).clamp_min(1e-4)
        th = th / th.norm(dim=1, keepdim=True).clamp_min(1e-4)
        m = (pt[:, k] > peak).float()
        l_d = l_d + ((1 - (ps * th).sum(1)) * m).sum() / m.sum().clamp_min(1.0)
    return l_h + 0.5 * l_d, dict(bce=float(l_h), dir=float(l_d))


''' + s[i1:]
j0 = s.index("    # E339 distillation: CWD of a map with itself")
j1 = s.index('    print("jmaps selftest ok; params"')
s = s[:j0] + '''    # E339 distillation. (1) the soft-target BCE is smallest at the teacher's own logits; (2) the teacher gets NO gradient although its forward kept the graph (the detach in `distill_loss` is tested, not the requires_grad flag);
    # (3) the heading term has zero gradient at a cell where the teacher sees no end and a gradient at its peak; (4) a student of width 32 follows a teacher of width 64: the loss falls by more than 20 % in 40 SGD steps
    teacher, student = JMaps(d=64), JMaps(d=32)
    tout = teacher(P)                                                                                       # gradients ENABLED: only the detach keeps the teacher out of the student's graph
    tout = dict(j1=tout["j1"] + torch.cat([torch.zeros(2, 2, 100, 100), 3.0 * torch.randn(2, 2, 100, 100), torch.zeros(2, 4, 100, 100)], 1))   # a teacher with structure: the heat logits are not all alike
    sj = dict(j1=tout["j1"].detach().clone())
    l0 = float(distill_loss(sj, tout)[0])
    assert l0 < float(distill_loss(dict(j1=sj["j1"] + torch.cat([torch.zeros(2, 2, 100, 100), torch.randn(2, 2, 100, 100), torch.zeros(2, 4, 100, 100)], 1)), tout)[0])
    assert l0 < float(distill_loss(dict(j1=sj["j1"] + torch.cat([torch.zeros(2, 2, 100, 100), 2.0 * torch.ones(2, 2, 100, 100), torch.zeros(2, 4, 100, 100)], 1)), tout)[0])
    ld0, opt = None, torch.optim.SGD(student.parameters(), lr=0.05)
    for it in range(40):
        sout = student(P)
        ld, di = distill_loss(sout, tout)
        ld0 = float(ld) if ld0 is None else ld0
        opt.zero_grad()
        ld.backward()
        opt.step()
    assert float(ld) < 0.8 * ld0 and all(p.grad is None for p in teacher.parameters()) and any(p.grad is not None for p in student.parameters()), (ld0, float(ld))
    assert set(di) == {"bce", "dir"}
    tj = torch.full((1, 8, 100, 100), -7.0)
    tj[0, 2, 30, 50] = 4.0
    tj[0, 4, :, :], tj[0, 5, :, :] = 0.0, 1.0
    sh = torch.zeros(1, 8, 100, 100, requires_grad=True)
    sh_in = sh + torch.cat([torch.zeros(1, 4, 100, 100), torch.ones(1, 1, 100, 100), torch.zeros(1, 3, 100, 100)], 1)                    # the student's heading (1, 0) against the teacher's (0, 1)
    distill_loss(dict(j1=sh_in), dict(j1=tj))[0].backward()
    g = sh.grad[0, 4:6].abs().sum(0)
    assert float(g[30, 50]) > 0 and float(g[10, 10]) == 0.0 and float(g[30, 51]) == 0.0, "the heading term must act at the teacher's peak only"
''' + s[j1:]
wr(p, s)

p = "training/train/jmaps_train.py"
s = rd(p)
s = sub(s, '''(channel-wise distillation, `jmaps.distill_loss`); the teacher is not saved and not trained")''', '''(soft-target heat BCE + heading cosine at the teacher's peaks, `jmaps.distill_loss`); the teacher is not saved and not trained")''')
s = sub(s, '''        tc = torch.load(a.teacher, map_location="cpu", weights_only=False)
''', '''        tc = torch.load(a.teacher, map_location="cpu", weights_only=False)
        assert not a.joint and not tc["args"].get("joint") and tc["args"].get("base") == a.base, f"the teacher {a.teacher} must read the same frozen pyramid (base {tc['args'].get('base')} against {a.base}, joint {tc['args'].get('joint')})"
''')
wr(p, s)
print("ok")
