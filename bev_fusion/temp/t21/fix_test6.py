p = "tests/test_stream_temporal.py"
s = open(p, encoding="utf-8").read()

def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a)
    s = s.replace(a, b)

rep('''for sub in ("tests", "training"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import model as M  # noqa: E402''', '''for sub in ("tests", "tests/support", "training", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import model as M  # noqa: E402
import synthetic_world as sw  # noqa: E402''')
rep('''from lift_pd import Grid  # noqa: E402
''', '''from lift_pd import Grid  # noqa: E402
from losses import BEVLoss, synthetic_gt  # noqa: E402
''')
rep('''# ------------------------------------------------------------------------------------------------ F mutants''', '''# ------------------------------------------------------------------------------------------------ H the trainer's own loop
print("H: through train.run_clip (two key frames, the state kept in the graph, the keyframe step of the poses)")


def with_gt(batch, grid):
    scene = sw.Scene([sw.Box(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), sw.Box(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0)],
                     lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": synthetic_gt(scene.step(-0.5), grid), "cur": synthetic_gt(scene, grid), "next": synthetic_gt(scene.step(0.5), grid)}
    gaps = {"prev": [(-1 / 12, 8.0, 0.05)] * 6, "next": [(1 / 12, 8.0, 0.05)] * 6}
    return dict(batch, gt=[side] * B, gap_steps=[gaps] * B, tokens=["a", "b"])


for kind in ("recurrent", "pair"):
    m = build(cam_buffer=kind, **MOTION).train()
    first = []
    orig_in = m._buffer_in
    m._buffer_in = lambda frames, state, step: first.append(orig_in(frames, state, step)) or first[-1]
    clip = [with_gt(mk(idx), m.grid) for idx in ([0, 1, 2], [3, 4, 5]) if kind == "recurrent" or True]
    if kind == "pair":                                               # the pair takes two frames: keyframes of (t-1, t)
        clip = [with_gt(mk(idx), m.grid) for idx in ([1, 2], [4, 5])]
    cfg = T.TrainConfig(seq_bptt=True, accum=1, flip_aug=0.0, gap_ego=True)
    m.zero_grad()
    acc, finite, _, totals = T.run_clip(m, BEVLoss(grid=m.grid), clip, 1.0, "cpu", "off", cfg, backward=True)
    g = [q.grad for q in m.cam_reduce.parameters()] + [q.grad for q in m.cam_temporal.head.parameters()]
    check(finite and len(totals) == 2 and all(x is not None and torch.isfinite(x).all() for x in g),
          f"{kind}: run_clip over two key frames is finite, the gradients of the lift and the temporal block exist")
    check(first[0] is None and (first[1] is not None) == (kind == "recurrent"),
          f"{kind}: the second key frame {'got' if kind == 'recurrent' else 'did not get'} the first one's buffer through the trainer's state and gap step")

''' + "# ------------------------------------------------------------------------------------------------ F mutants")
open(p, "w", encoding="utf-8", newline="").write(s)
