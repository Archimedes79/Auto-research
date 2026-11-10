"""One buffered frame instead of the window: `ModelConfig.cam_buffer` 'pair' / 'recurrent' (card t21, Mirko's D51); CPU, about 2 min, no data.

  A  pair: one stream step (ONE frame + state["buf"]["raw"] of the previous call, the ego chain as `step`) = the batch of the two frames, 1e-5,
     for dt -0.5 / 0 / +0.5, also with the motion block (it reads the two raw BEVs).
  B  recurrent: three stream steps from cold = the batch unroll of the three frames; and across keyframes (the trainer's way): a batch that gets
     the previous batch's state and the keyframe gap step = the stream over the same six frames; the carried buffer changes the output; a
     batch never reads a buffer of another call without a step (pair: never).
  C  gradient: in the recurrent batch the loss reaches the oldest frame's lift through the carried fused state (not only through the world
     memory), and the first keyframe's lift through the buffer carried to the next keyframe; every gradient is finite.
  D  cold start in the stream: finite, equal to the window model with one frame (no motion block); with the motion block the current frame stands
     in as its older frame with a zero step; a buffered stream step without `step` and a 'pair' batch of three frames are refused;
     `advance` (a stream frame that needs no output: lift + temporal block only) returns the state of the full step.
  E  nothing else changes: cam_buffer 'window' (default and named) gives the camera BEV, motion field and state of the code before t21 (the
     old composition is kept in this file), the three kinds have the same state_dict keys and shapes as before (golden digest of the tiny
     'dual' model of test_lane_branch_dual), `check_config` rules.
  H  the trainer's own loop: `train.run_clip` over two key frames hands the first one's buffer to the second (recurrent), never in a pair batch;
     the gradients of the lift and the temporal block exist.
  F  mutants: eight planted bugs in the new code (carry ignored, wrong frame as the previous one, wrong tensor in the buffer, step dropped, gap
     step not reduced to the oldest frame, carry cut by a detach, a batch that reads a buffer, motion block on the oldest frame) each fail a check.
  G  the ego step as ONE composed triple (train.gap_ego_motion) against the per-sweep chain on a turning ego: the position error in cells (measured
     and bounded), the reason a stream caller should pass the chain.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_stream_temporal.py"""
import hashlib
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("tests", "tests/support", "training", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import model as M  # noqa: E402
import synthetic_world as sw  # noqa: E402
import splat as S  # noqa: E402
import train as T  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from lift_pd import Grid  # noqa: E402
from losses import BEVLoss, synthetic_gt  # noqa: E402
from model import ModelConfig  # noqa: E402
from motion_block import block_inputs  # noqa: E402
from precompute import Precompute  # noqa: E402
from temporal_fusion import TemporalFusion  # noqa: E402
from model_gates import refuser  # noqa: E402
from test_slots import batch_for  # noqa: E402

torch.set_num_threads(2)
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}                 # 32 x 32 cells
TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24,
            trunk_width=0.25, p_drop=0.0)
MOTION = dict(motion_block=True, box_vel_from_field=True)
DTS = (-0.5, 0.0, 0.5)
B, NF, TOL = 2, 6, 1e-5
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------------------------------------ the world: NF consecutive frames
CFG0 = ModelConfig(**TINY)
BASE = batch_for(CFG0, B=B, Fn=NF)                                   # random features per frame, one camera
SEGS = [[(torch.full((B,), 1 / 12), torch.tensor([8.0 + 0.5 * i + 0.3 * j, 12.0 - 0.4 * i - 0.2 * j]),
          torch.tensor([0.30 - 0.05 * i + 0.02 * j, -0.20 + 0.04 * i - 0.03 * j])) for j in range(2)] for i in range(NF - 1)]   # frame i -> i + 1, two sweeps
VG, WG = torch.tensor([9.0, 11.0]), torch.tensor([0.25, -0.15])      # the gap mean ego motion of the keyframe step


def mk(idx):
    """the batch of the frames `idx` (chronological), stored newest first as the collate does; steps[k] = the sweeps of frame k to the newest"""
    order = idx[::-1]
    b = dict(BASE)
    for k in ("K", "T_cam_car", "prob", "ctx"):
        b[k] = BASE[k][:, order]
    b["steps"] = [sum((SEGS[i] for i in range(o, idx[-1])), []) for o in order]
    return b


def build(**kw):
    torch.manual_seed(0)
    m = CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw))).eval()
    with torch.no_grad():                                            # zero-start layers get values: every path carries a signal
        for q in m.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 0.05)
    return m


def run(m, batch, state=None, step=None):
    with torch.no_grad():
        return m(batch, dts=DTS, state=state, step=step)


def flat(res, buf=True):
    out = {"features": res["features"]}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    out[f"{dt}.{k}.{kk}"] = t
    for k, v in res["state"].items():
        for kk, t in (v or {}).items():
            if buf or k != "buf":
                out[f"state.{k}.{kk}"] = t
    return out


def gap(a, b, buf=True):
    """the largest absolute difference of every tensor of two results (outputs, features, states; buf=False: not the buffer)"""
    fa, fb = flat(a, buf), flat(b, buf)
    assert fa.keys() == fb.keys(), set(fa) ^ set(fb)
    return max(float((fa[k] - fb[k]).abs().max()) for k in fa)


def stream(m, idx):
    """one call per frame of `idx` (consecutive), the state carried; every call after the first gets the chain from the frame before"""
    res, st = [], None
    for n, i in enumerate(idx):
        res.append(run(m, mk([i]), st, SEGS[i - 1] if n else None))
        st = res[-1]["state"]
    return res


KEYS = (("plain", {}), ("motion", MOTION))


def pair_checks(tag, kw):
    p = build(cam_buffer="pair", **kw)
    r0 = run(p, mk([0]))                                              # the previous call: frame 0, cold
    w = build(**kw)                                                   # the window model, the same weights: two frames are its window with K = 1
    return {f"A pair stream = batch of two frames ({tag})": gap(run(p, mk([1]), r0["state"], SEGS[0]), run(p, mk([0, 1]), r0["state"], SEGS[0])) < TOL,
            f"A2 pair batch of two frames = the window model on the same two frames ({tag})": gap(run(p, mk([0, 1])), run(w, mk([0, 1])), buf=False) == 0}


def recurrent_checks(tag, kw):
    r = build(cam_buffer="recurrent", **kw)
    ok = {f"B1 recurrent stream of three frames = batch unroll ({tag})": gap(stream(r, [0, 1, 2])[2], run(r, mk([0, 1, 2]))) < TOL}
    if kw:                                                            # the motion block reads the two raw BEVs, the frame before the current one
        ref = run(build(**kw), mk([1, 2]))["out"][0.0]
        got = run(r, mk([0, 1, 2]))["out"][0.0]
        ok[f"B6 the motion field of the recurrent batch = the window model's on the last two frames ({tag})"] =             all(torch.equal(got[k], ref[k]) if torch.is_tensor(got[k]) else all(torch.equal(got[k][j], ref[k][j]) for j in ref[k]) for k in ("v_hat", "v_levels"))
    s = stream(r, [0, 1, 2]) + [None] * 3                             # frames 3 - 5 follow; the step to frame 3 is the keyframe step less its own chain
    dt_o = torch.full((B,), 0.5) - sum((x[0] for x in SEGS[3] + SEGS[4]), torch.zeros(B))
    s[3] = run(r, mk([3]), s[2]["state"], [(dt_o, VG, WG)])
    s[4] = run(r, mk([4]), s[3]["state"], SEGS[3])
    s[5] = run(r, mk([5]), s[4]["state"], SEGS[4])
    k1 = run(r, mk([0, 1, 2]))
    k2 = run(r, mk([3, 4, 5]), k1["state"], (torch.full((B,), 0.5), VG, WG))
    ok[f"B2 keyframe batch with the previous state and the gap step = the stream of six frames ({tag})"] = gap(s[5], k2) < TOL
    cold = run(r, mk([3, 4, 5]), {k: v for k, v in k1["state"].items() if k != "buf"}, (torch.full((B,), 0.5), VG, WG))
    ok[f"B3 the carried buffer matters ({tag})"] = gap(k2, cold) > 1e-3
    return ok


def buffer_rules():
    bogus = {"raw": torch.randn(B, 16, 32, 32), "fused": torch.randn(B, 16, 32, 32)}
    r = build(cam_buffer="recurrent")
    p = build(cam_buffer="pair")
    r0 = run(p, mk([0]))
    return {"B4 recurrent batch without a step ignores a buffer (cold)": gap(run(r, mk([0, 1, 2]), {"buf": bogus}), run(r, mk([0, 1, 2]))) < 1e-7,
            "B5 pair batch never reads a buffer": gap(run(p, mk([0, 1]), dict(r0["state"], buf=bogus), SEGS[0]), run(p, mk([0, 1]), r0["state"], SEGS[0])) < 1e-7}


def advance_checks():
    ok = {}
    for kind in ("pair", "recurrent"):
        m = build(cam_buffer=kind, **MOTION)
        r0 = run(m, mk([0]))
        with torch.no_grad():
            cold, warm = m.advance(mk([0])), m.advance(mk([1]), r0["state"], SEGS[0])
        full = run(m, mk([1]), r0["state"], SEGS[0])["state"]
        same = lambda a, b: all(torch.equal(a[k][kk], b[k][kk]) for k in ("cam", "buf") for kk in b[k])      # noqa: E731
        ok[f"D advance (lift + temporal block only) gives the state of the full step, cold and buffered ({kind})"] = same(cold, r0["state"]) and same(warm, full)
    return ok


def core_checks():
    """the equalities and rules the mutants must break: name -> bool (an exception counts as a failed group)"""
    ok = {}
    for fn, *args in [(pair_checks, t, k) for t, k in KEYS] + [(recurrent_checks, t, k) for t, k in KEYS] + [(buffer_rules,), (advance_checks,), (grad_checks,)]:
        try:
            ok.update(fn(*args))
        except Exception as e:                                        # noqa: BLE001 - a planted bug that crashes is caught as well
            ok[f"{fn.__name__}{args[:1]} raised {type(e).__name__}"] = False
    return ok


# ------------------------------------------------------------------------------------------------ C gradient
def loss_of(res):
    g = torch.Generator().manual_seed(3)
    return res["features"].square().mean() + (res["out"][0.0]["sem"] * torch.randn(res["out"][0.0]["sem"].shape, generator=g)).mean()


def grads(m, carry_cut=False):
    """the recurrent batch of frames 0 - 2 in train mode: (gradient of the loss on the OLDEST frame's context features, on cam_reduce's weight);
    carry_cut: the reference whose block sees the carried fused state as a constant"""
    m.train()
    b = mk([0, 1, 2])
    ctx = b["ctx"].clone().requires_grad_(True)
    b["ctx"] = ctx
    orig = TemporalFusion.forward
    if carry_cut:
        TemporalFusion.forward = lambda self, cur, past=(), *a, **k: orig(self, cur, [(x.detach(), s) for x, s in past], *a, **k)
    try:
        m.zero_grad()
        loss_of(m(b, dts=(0.0, 0.5))).backward()
    finally:
        TemporalFusion.forward = orig
        m.eval()
    assert all(q.grad is None or torch.isfinite(q.grad).all() for q in m.parameters()), "a gradient is not finite"
    return ctx.grad[:, 2].clone(), m.cam_reduce[0].weight.grad.clone()


def grad_checks():
    ok = {}
    m = build(cam_buffer="recurrent")
    g_old, g_red = grads(m)
    c_old, c_red = grads(m, carry_cut=True)
    rel = float((g_old - c_old).norm() / g_old.norm())
    ok["C1 the oldest frame's lift gets a gradient through the carried fused state"] = float(g_old.norm()) > 0 and rel > 1e-2
    ok["C2 cam_reduce's gradient differs from the carry-cut run"] = float((g_red - c_red).norm() / g_red.norm()) > 1e-3
    m.train()                                                         # keyframe 1 -> 2: the loss of the second reaches the first through the buffer
    b1, b2 = mk([0, 1, 2]), mk([3, 4, 5])
    ctx1 = b1["ctx"].clone().requires_grad_(True)
    b1["ctx"] = ctx1
    r1 = m(b1, dts=(0.0,))
    st = r1["state"]
    gg = []
    for state in (st, dict(st, buf={k: v.detach() for k, v in st["buf"].items()})):
        ctx1.grad = None
        loss_of(m(b2, dts=(0.0, 0.5), state=state, step=(torch.full((B,), 0.5), VG, WG))).backward(retain_graph=True)
        gg.append(ctx1.grad.clone())
    m.eval()
    ok["C3 the first keyframe's lift gets a gradient through the carried buffer"] = float((gg[0] - gg[1]).norm() / gg[0].norm()) > 1e-3
    return ok


# ------------------------------------------------------------------------------------------------ A - D: baseline
print("A - C: stream = batch, buffer rules, gradient")
t0 = time.time()
base = core_checks()
for k, v in base.items():
    check(v, k)

print("D: cold start")
for kind in ("pair", "recurrent"):
    w, m = build(), build(cam_buffer=kind)
    m.load_state_dict(w.state_dict())
    check(gap(run(m, mk([2])), run(w, mk([2])), buf=False) == 0, f"{kind}: cold stream step = the window model with one frame (no past, no motion block), bit for bit")
    mm = build(cam_buffer=kind, **MOTION)
    c = flat(run(mm, mk([2])))
    check(all(torch.isfinite(t).all() for t in c.values()) and float(c["0.0.v_hat."].abs().max()) < 1e3,
          f"{kind} + motion block: cold start (the current frame as its own older frame, zero step) is finite")
    s1 = run(mm, mk([3]), run(mm, mk([2]))["state"], SEGS[2])
    check(all(torch.isfinite(t).all() for t in flat(s1).values()) and "buf" in s1["state"], f"{kind} + motion block: the first buffered step is finite")


def refuses(call):
    try:
        call()
        return False
    except ValueError:
        return True


mm = build(cam_buffer="recurrent", **MOTION)
check(refuses(lambda: run(mm, mk([3]), run(mm, mk([2]))["state"])), "a buffered stream step without `step` is refused")
check(refuses(lambda: run(build(cam_buffer="pair"), mk([0, 1, 2]))), "a 'pair' batch of three frames is refused")
check(refuses(lambda: build().advance(mk([0]))), "advance on a window model is refused")

# ------------------------------------------------------------------------------------------------ E
print("E: nothing else changes")


def old_composition(m, batch, state, step):
    """the camera section of forward() before t21: window path + the motion block on the OLDEST frame"""
    frames = m.camera_bevs(batch)
    steps = [[tuple(torch.as_tensor(t, dtype=frames.dtype) for t in s) for s in f] for f in batch["steps"]]
    obs = Precompute.from_model(m).camera(dict(batch)).to(frames)
    cam, cam_state = M.BEVFusionModel._branch_temporal(m.cam_temporal, frames, steps, (state or {}).get("cam"), step, obs)
    v = m.motion_block(*block_inputs(frames, steps, batch["radar"], m.grid))[-1] if m.motion_block is not None else None
    return cam, cam_state, v


for named in (False, True):
    m = build(**(dict(cam_buffer="window") if named else {}), **MOTION)
    caps = []
    h = m.fusion.register_forward_hook(lambda mod, inp, out: caps.append(inp[0]))
    b = mk([1, 2, 3])
    r1 = run(m, b)
    r2 = run(m, b, r1["state"], (torch.tensor([0.5, 0.45]), torch.tensor([8.0, 10.0]), torch.tensor([0.05, -0.1])))
    h.remove()
    cam1, st1, v1 = old_composition(m, b, None, None)
    cam2, st2, v2 = old_composition(m, b, r1["state"], (torch.tensor([0.5, 0.45]), torch.tensor([8.0, 10.0]), torch.tensor([0.05, -0.1])))
    check(torch.equal(caps[0], cam1) and torch.equal(caps[1], cam2) and all(torch.equal(st2[k], r2["state"]["cam"][k]) for k in st2)
          and torch.equal(r1["out"][0.0]["v_hat"], v1) and torch.equal(r2["out"][0.0]["v_hat"], v2) and "buf" not in r1["state"],
          f"window ({'named' if named else 'default'}): camera BEV, memory state and motion field (oldest frame) = the code before t21, no buffer in the state")
m = build(**MOTION)
check(float((run(m, mk([1, 2, 3]))["out"][0.0]["v_hat"] - run(build(cam_buffer="recurrent", **MOTION), mk([1, 2, 3]))["out"][0.0]["v_hat"]).abs().max()) > 1e-4,
      "the window's motion block (oldest frame, 0.333 s) and the buffered kinds' (the frame before, 0.167 s) read different frames")

TINY_DUAL = dict(TINY, camera_features="dual")
digests = set()
for kind in M.CAM_BUFFERS:
    torch.manual_seed(0)
    rows = [f"{k} {tuple(v.shape)}" for k, v in CachedBEVFusionModel(ModelConfig(**dict(TINY_DUAL, cam_buffer=kind))).state_dict().items()]
    digests.add((len(rows), hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]))
check(digests == {(223, "0f169833b881bec8")}, f"cam_buffer window / pair / recurrent: the state_dict keys and shapes of before, no new parameter ({digests})")


refused = refuser(TINY)


check(refused(cam_buffer="stack") and refused(cam_buffer="pair", cam_temporal="none") and refused(cam_buffer="recurrent", cam_temporal="ego_gru")
      and not refused(cam_buffer="pair") and not refused(cam_buffer="recurrent") and not refused(cam_buffer="window", cam_temporal="none"),
      "check_config: an unknown value, and 'pair' / 'recurrent' without the window-memory block are refused; 'window' is free")

# ------------------------------------------------------------------------------------------------ H the trainer's own loop
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
    clip = [with_gt(mk(idx), m.grid) for idx in (([0, 1, 2], [3, 4, 5]) if kind == "recurrent" else ([1, 2], [4, 5]))]
    cfg = T.TrainConfig(seq_bptt=True, accum=1, flip_aug=0.0, gap_ego=True)
    m.zero_grad()
    acc, finite, _, totals = T.run_clip(m, BEVLoss(grid=m.grid), clip, 1.0, "cpu", "off", cfg, backward=True)
    g = [q.grad for q in m.cam_reduce.parameters()] + [q.grad for q in m.cam_temporal.head.parameters()]
    check(finite and len(totals) == 2 and all(x is not None and torch.isfinite(x).all() for x in g),
          f"{kind}: run_clip over two key frames is finite, the gradients of the lift and the temporal block exist")
    check(first[0] is None and (first[1] is not None) == (kind == "recurrent"),
          f"{kind}: the second key frame {'got' if kind == 'recurrent' else 'did not get'} the first one's buffer through the trainer's state and gap step")

# ------------------------------------------------------------------------------------------------ F mutants
print("F: planted bugs")


def mutants():
    prev = M.BEVFusionModel._previous

    def wrong_previous(self, frames, steps, buf, step):
        pair, ps, warm = prev(self, frames, steps, buf, step)
        return torch.stack([pair[:, 0], pair[:, 0]], 1), ps, warm            # the current frame stands in for the previous one

    def oldest_for_motion(self, frames, steps, buf, step):
        pair, ps, warm = prev(self, frames, steps, buf, step)
        return (frames[:, [0, -1]], [steps[0], steps[-1]], True) if frames.shape[1] > 1 else (pair, ps, warm)

    return {
        "the carry is ignored (every call cold)": [(M.BEVFusionModel, "_buffer_in", lambda self, frames, state, step: None)],
        "the current frame stands in for the previous one": [(M.BEVFusionModel, "_previous", wrong_previous)],
        "the raw and fused tensors are swapped in the buffer": [(M.BEVFusionModel, "_next_buffer", staticmethod(
            lambda frames, fused, recurrent: {"raw": fused, "fused": frames[:, 0]} if recurrent else {"raw": fused}))],
        "the ego step is dropped": [(M, "as_chain", lambda step, like=None: [])],
        "the gap step is not reduced to the oldest frame": [(M.BEVFusionModel, "_step_to_oldest", staticmethod(lambda gap, steps, frames: gap))],
        "a batch reads any buffer": [(M.BEVFusionModel, "_buffer_in", lambda self, frames, state, step: state.get("buf"))],
        "the motion block reads the oldest frame in a batch": [(M.BEVFusionModel, "_previous", oldest_for_motion)],
    }


def detached_carry():
    orig = TemporalFusion.forward
    return [(TemporalFusion, "forward", lambda self, cur, past=(), *a, **k: orig(self, cur, [(x.detach(), s) for x, s in past], *a, **k))]


planted = dict(mutants(), **{"the carried fused state is cut by a detach": detached_carry()})
for name, patches in planted.items():
    saved = [(o, a, vars(o)[a]) for o, a, _ in patches]              # vars(): a staticmethod stays one
    for o, a, f in patches:
        setattr(o, a, f)
    try:
        try:
            res = core_checks()
            died = [k for k, v in res.items() if not v]
        except Exception as e:                                       # noqa: BLE001 - a mutant that crashes a check is caught as well
            died = [f"raised {type(e).__name__}"]
    finally:
        for o, a, f in saved:
            setattr(o, a, f)
    check(bool(died), f"mutant killed: {name} -> {died[0] if died else 'SURVIVED'}{' (+%d)' % (len(died) - 1) if len(died) > 1 else ''}")
check(all(core_checks().values()), "after the mutants the baseline passes again")

# ------------------------------------------------------------------------------------------------ G
print("G: one composed triple against the per-sweep chain (turning ego)")
grid = Grid()
pts = S.grid_points(grid, dtype=torch.float64)[None]
worst = {}
for n in (2, 6):
    for v0, w0 in ((10.0, 0.2), (10.0, 0.5), (15.0, -0.8), (5.0, 1.2)):
        chain = [(1 / 12, v0 + 0.3 * i, w0 * (1 + 0.1 * i)) for i in range(n)]
        err = float(((S.chain_positions(pts, chain) - S.splat_positions(pts, *T.gap_ego_motion(chain))).norm(dim=-1) / grid.res).max())
        worst[n] = max(worst.get(n, 0.0), err)
print(f"  worst position error over the grid, omega up to 1.2 rad/s: {worst[2]:.4f} cells for a 2-sweep step (0.167 s, one frame), "
      f"{worst[6]:.4f} cells for the 6-sweep keyframe step (0.5 s)")
check(worst[2] < 0.01 and worst[6] < 0.2, "a single composed triple is within 0.01 cell (one frame step) / 0.2 cell (keyframe step) of the chain; the stream takes the chain")

print(f"\n{'FAILED: ' + str(len(fails)) + ' check(s)' if fails else 'test_stream_temporal passed'} ({time.time() - t0:.0f} s)")
sys.exit(1 if fails else 0)
