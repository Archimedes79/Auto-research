"""One radar sweep per step and the radar past in the feature state: `ModelConfig.radar_per_frame` (card t25, the owner's D91); tiny cached model, CPU, no data, about 3 min.

  A  stream = batch: three one-frame stream steps from cold = the batch unroll of the three frames, and across key frames (a batch that gets the previous batch's state and
     the keyframe gap step) = the stream over the same six frames, each frame with its own sweep (radar_seq); output, features, radar state and buffer to 1e-5. The
     carried radar buffer matters. `advance` (lift + temporal blocks, no output) gives the state of the full step, cold and buffered.
  B  step 0: the radar memory enters as a residual from the identity (radar features + rad_mix(block output), rad_mix zero): the output of the memory arm equals the
     output of the same network without the memory (rad_temporal none), bit for bit, in a key frame and in the next one, and the carried radar buffer is the radar
     block's output on the newest sweep; with radar_kind delta_slots and the motion block, radar_side (features and raster) is the radar block's on radar_seq[0] and the
     no-memory arm's; with rad_mix awake the outputs differ.
  C  gradient (rad_mix awake): the radar block, every tensor of the radar memory block (velocity head, relevance, tau, fusion) and rad_mix get a finite non-zero
     gradient; the radar block's gradient differs from a run with the carried fused radar BEV cut (the recursion passes it), also across key frames.
  D  the write mask of the radar memory is made per frame from THAT frame's sweep (radar_static with the gate, radar_hits without), oldest frame first; the radar block
     gets the camera block's ego step and past chains at every frame (v and omega, two key frames).
  E  nothing else changes: radar_per_frame off ignores radar_seq (outputs bit for bit), the on-network has the off-network's tensors plus rad_mix.*; check_config and the
     forward refuse the misuse (no recursive camera, no window-memory radar block, a batch without radar_seq or with too few sets).
  F  plumbing of batch["radar_seq"]: interface.check_inputs knows it, train.to_device moves every set, augment.flip_batch mirrors every set (y, psi) of the flipped
     samples only, and the trainer's own loop (train.run_clip, flip_aug 1.0, loss stubbed) hands the model the mirrored sets; Precompute.radar_mask = the old maps.
     (The real-data part - radar_seq[0] = the radar_window 0.09 radar, each frame's newest sweep, about 54 points in the grid per key frame - is in the self-test of
     training/nuscenes_dataset.py.)
  G  mutants: nine planted bugs (radar carry ignored, radar frames in the wrong order, masks in the wrong order, radar state not carried, fused radar BEV cut by a
     detach, radar side loss without features / from the oldest frame, wrong sign of v / of omega in the radar's ego chains) each fail a check; the last four were
     found unguarded by the review of 2026-11-14.
  H  configs/t25_*.json: t25_ctl is t21_pair4.json with batch 2 / accum 3, t25_rec_r is t21_rec.json, the other two add exactly the changes of the card; their keys are TrainConfig
     fields and their model sections pass check_config.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_radar_recursive.py"""
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for sub in ("tests", "tests/support", "training", ".."):
    sys.path.insert(0, os.path.join(ROOT, sub))
import model as M  # noqa: E402
import train as T  # noqa: E402
from augment import flip_batch  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from interface import RADAR_FIELDS, check_inputs  # noqa: E402
from model import ModelConfig, check_config  # noqa: E402
from model_gates import refuser  # noqa: E402
from precompute import Precompute, radar_hits, radar_static  # noqa: E402
from temporal_fusion import TemporalFusion  # noqa: E402
from test_slots import batch_for  # noqa: E402

torch.set_num_threads(2)
GRID = {"x_max": 16.0, "y_min": -8.0, "y_max": 8.0}                 # 32 x 32 cells
TINY = dict(pretrained="", trunk_pretrained="", lane_pretrained="", grid_kwargs=GRID, oversample=1, c_cam=16, c_rad=8, c_fuse=24,
            trunk_width=0.25, p_drop=0.0)
MEM = dict(cam_buffer="recurrent", rad_temporal="window_memory", rad_static_gate=True, radar_per_frame=True)
MOTION = dict(motion_block=True, box_vel_from_field=True)
DTS = (-0.5, 0.0, 0.5)
B, NF, TOL = 2, 6, 1e-5
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ------------------------------------------------------------------------------------------------ the world: NF consecutive frames, each with its own sweep
BASE = batch_for(ModelConfig(**TINY), B=B, Fn=NF)                    # random features per frame, one camera
SEGS = [[(torch.full((B,), 1 / 12), torch.tensor([8.0 + 0.5 * i + 0.3 * j, 12.0 - 0.4 * i - 0.2 * j]),
          torch.tensor([0.30 - 0.05 * i + 0.02 * j, -0.20 + 0.04 * i - 0.03 * j])) for j in range(2)] for i in range(NF - 1)]   # frame i -> i + 1, two sweeps
VG, WG = torch.tensor([9.0, 11.0]), torch.tensor([0.25, -0.15])      # the gap mean ego motion of the keyframe step


def sweep(i):
    """the radar set of frame i: 48 detections inside the tiny grid, other points in every frame, Doppler mostly near 0"""
    g = torch.Generator().manual_seed(100 + i)
    n = 48
    d = {f: torch.randn(n, generator=g) for f in RADAR_FIELDS}
    d.update(x=torch.rand(n, generator=g) * 15.5, y=torch.rand(n, generator=g) * 15.5 - 7.75, dt=torch.rand(n, generator=g) * 0.09, b=torch.arange(n) % B)
    d["vr"] = torch.randn(n, generator=g) * 0.7
    return d


RAD = [sweep(i) for i in range(NF)]


def mk(idx):
    """the batch of the frames `idx` (chronological), stored newest first as the collate does; steps[k] = the sweeps of frame k to the newest;
    radar_seq[k] = frame k's own radar set, radar = the newest one"""
    order = idx[::-1]
    b = dict(BASE)
    for k in ("K", "T_cam_car", "prob", "ctx"):
        b[k] = BASE[k][:, order]
    b["steps"] = [sum((SEGS[i] for i in range(o, idx[-1])), []) for o in order]
    b["radar"], b["radar_seq"] = RAD[idx[-1]], [RAD[i] for i in order]
    return b


def build(zero_mix=False, **kw):
    """a tiny cached model in eval mode; zero-start layers get values so every path carries a signal (rad_mix stays zero with zero_mix)"""
    torch.manual_seed(0)
    m = CachedBEVFusionModel(ModelConfig(**dict(TINY, **kw))).eval()
    with torch.no_grad():
        for n, q in m.named_parameters():
            if q.requires_grad and q.abs().max() == 0 and not (zero_mix and n.startswith("rad_mix")):
                q.normal_(0, 0.05)
    return m


def run(m, batch, state=None, step=None):
    with torch.no_grad():
        return m(batch, dts=DTS, state=state, step=step)


def flat(res, state=True):
    out = {"features": res["features"]}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    out[f"{dt}.{k}.{kk}"] = t
    for k, v in (res["state"].items() if state else []):
        for kk, t in (v or {}).items():
            out[f"state.{k}.{kk}"] = t
    return out


def gap(a, b, state=True):
    """the largest absolute difference of every tensor of two results (outputs, features, states)"""
    fa, fb = flat(a, state), flat(b, state)
    assert fa.keys() == fb.keys(), set(fa) ^ set(fb)
    return max(float((fa[k] - fb[k]).abs().max()) for k in fa)


def stream(m, idx):
    """one call per frame of `idx` (consecutive), the state carried; every call after the first gets the chain from the frame before"""
    res, st = [], None
    for n, i in enumerate(idx):
        res.append(run(m, mk([i]), st, SEGS[i - 1] if n else None))
        st = res[-1]["state"]
    return res


# ------------------------------------------------------------------------------------------------ A stream = batch
def stream_checks():
    r = build(**MEM)
    ok = {"A1 stream of three frames (each with its own sweep) = batch unroll, radar state and buffer included": gap(stream(r, [0, 1, 2])[2], run(r, mk([0, 1, 2]))) < TOL}
    s = stream(r, [0, 1, 2]) + [None] * 3                             # frames 3 - 5 follow; the step to frame 3 is the keyframe step less its own chain
    dt_o = torch.full((B,), 0.5) - sum((x[0] for x in SEGS[3] + SEGS[4]), torch.zeros(B))
    s[3] = run(r, mk([3]), s[2]["state"], [(dt_o, VG, WG)])
    s[4] = run(r, mk([4]), s[3]["state"], SEGS[3])
    s[5] = run(r, mk([5]), s[4]["state"], SEGS[4])
    k1 = run(r, mk([0, 1, 2]))
    k2 = run(r, mk([3, 4, 5]), k1["state"], (torch.full((B,), 0.5), VG, WG))
    ok["A2 key frame batch with the previous state and the gap step = the stream of six frames"] = gap(s[5], k2) < TOL
    cut = dict(k1["state"], buf=dict(k1["state"]["buf"], rad_fused=torch.zeros_like(k1["state"]["buf"]["rad_fused"])), rad=None)
    ok["A3 the carried radar buffer and state matter"] = gap(k2, run(r, mk([3, 4, 5]), cut, (torch.full((B,), 0.5), VG, WG))) > 1e-4
    first = run(r, mk([0]))["state"]
    with torch.no_grad():
        cold, warm = r.advance(mk([0])), r.advance(mk([1]), first, SEGS[0])
    full = run(r, mk([1]), first, SEGS[0])["state"]
    same = lambda a, b: all(torch.equal(a[k][kk], b[k][kk]) for k in ("cam", "rad", "buf") for kk in b[k])      # noqa: E731
    ok["A4 advance (lift + temporal blocks, no output) gives the state of the full step, cold and buffered"] = same(cold, first) and same(warm, full)
    return ok


# ------------------------------------------------------------------------------------------------ C gradient
def loss_of(res):
    g = torch.Generator().manual_seed(3)
    return res["features"].square().mean() + (res["out"][0.0]["sem"] * torch.randn(res["out"][0.0]["sem"].shape, generator=g)).mean()


def radar_grad(m, cut=False):
    """the recurrent batch of frames 0 - 2 in train mode: the gradient of the loss on the radar block's first tensor; cut: the radar memory block sees the carried
    fused radar BEV as a constant"""
    m.train()
    if cut:
        orig = m.rad_temporal.forward
        m.rad_temporal.forward = lambda cur, past=(), *a, **k: orig(cur, [(x.detach(), s) for x, s in past], *a, **k)
    try:
        m.zero_grad()
        loss_of(m(mk([0, 1, 2]), dts=(0.0, 0.5))).backward()
    finally:
        if cut:
            del m.rad_temporal.forward
        m.eval()
    return next(m.radar.parameters()).grad.clone()


def grad_checks():
    ok = {}
    m = build(**MEM)
    g = radar_grad(m)
    bad = [n for n, q in m.named_parameters() if n.startswith(("radar.", "rad_temporal.", "rad_mix.")) and q.requires_grad
           and not (q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0)]
    ok["C1 the radar block, every tensor of the radar memory block and rad_mix get a finite non-zero gradient" + (f" (none: {bad})" if bad else "")] = not bad
    c = radar_grad(m, cut=True)
    ok["C2 the radar block's gradient differs from the run with the carried fused radar BEV cut"] = float((g - c).norm() / g.norm()) > 1e-3
    m.train()                                                         # key frame 1 -> 2: the loss of the second reaches the first radar through the buffer
    st = m(mk([0, 1, 2]), dts=(0.0,))["state"]
    w, gg = next(m.radar.parameters()), []
    for state in (st, T.detach_tree(st)):
        w.grad = None
        loss_of(m(mk([3, 4, 5]), dts=(0.0, 0.5), state=state, step=(torch.full((B,), 0.5), VG, WG))).backward(retain_graph=True)
        gg.append(w.grad.clone())
    m.eval()
    ok["C3 the first key frame's radar block gets a gradient through the carried buffer"] = float((gg[0] - gg[1]).norm() / gg[0].norm()) > 1e-3
    return ok


# ------------------------------------------------------------------------------------------------ D what the radar block gets at each frame
def tap(block, log):
    """record (step, the chains of the past entries, write mask) of every call of a temporal block; `del block.forward` takes the hook away"""
    orig = block.forward
    block.forward = lambda cur, past=(), state=None, step=None, obs_mass=None: log.append((step, [s for _, s in past], obs_mass)) or orig(cur, past, state, step, obs_mass)


def same_chain(a, b):
    """two ego chains (lists of (dt, v, omega) tensor triples), or lists of such chains, hold the same tensors"""
    if torch.is_tensor(a):
        return torch.equal(a, b)
    return len(a) == len(b) and all(same_chain(x, y) for x, y in zip(a, b))


def mask_checks():
    """over two key frames (six calls of each block): the radar block's write mask is that of its own sweep, and its ego step and the chains of its past are the camera block's"""
    ok = {}
    for gate in (True, False):
        m = build(**dict(MEM, rad_static_gate=gate))
        rad, cam = [], []
        tap(m.rad_temporal, rad)
        tap(m.cam_temporal, cam)
        k1 = run(m, mk([0, 1, 2]))
        run(m, mk([3, 4, 5]), k1["state"], (torch.full((B,), 0.5), VG, WG))
        del m.rad_temporal.forward, m.cam_temporal.forward
        want = [(radar_static(RAD[i], B, m.grid, 1.0, 1) if gate else radar_hits(RAD[i], B, m.grid, 1)) for i in range(6)]
        kind = "radar_static" if gate else "radar_hits"
        ok[f"D write mask per frame from that frame's own sweep, oldest first ({kind})"] = \
            len(rad) == 6 and all(torch.equal(a[2], b) for a, b in zip(rad, want)) and not torch.equal(want[0], want[2])
        ok[f"D the radar block gets the camera block's ego step and past chains at every frame, v and omega ({kind})"] = \
            len(cam) == 6 and all(same_chain(a[0], b[0]) and same_chain(a[1], b[1]) for a, b in zip(rad, cam)) \
            and any(len(a[0]) and float(a[0][0][1].abs().sum()) > 0 and float(a[0][0][2].abs().sum()) > 0 for a in rad)
    return ok


# ------------------------------------------------------------------------------------------------ B step 0
def twins(**kw):
    """(the memory arm with rad_mix zero, the same network without the memory that takes every tensor of it that it has)"""
    mem, ref = build(zero_mix=True, **kw, **MEM), build(**kw, cam_buffer="recurrent")
    miss = ref.load_state_dict({k: v for k, v in mem.state_dict().items() if k in ref.state_dict()}, strict=False)
    assert not miss.missing_keys and not miss.unexpected_keys, miss
    return mem, ref


def same(a, b):
    """features and every output of two results equal bit for bit (states not compared: the memory arm has more)"""
    fa, fb = flat(a, False), flat(b, False)
    return fa.keys() == fb.keys() and all(torch.equal(fa[k], fb[k]) for k in fa)


def identity_checks():
    ok = {}
    step = (torch.full((B,), 0.5), VG, WG)
    mem, ref = twins()
    k1m, k1r = run(mem, mk([0, 1, 2])), run(ref, mk([0, 1, 2]))
    k2m, k2r = run(mem, mk([3, 4, 5]), k1m["state"], step), run(ref, mk([3, 4, 5]), k1r["state"], step)
    ok["B1 rad_mix zero: features and every output of the memory arm = the arm without the memory, bit for bit (key frame and the next one)"] = same(k1m, k1r) and same(k2m, k2r)
    with torch.no_grad():
        e2, e5 = mem.radar(RAD[2], B), mem.radar(RAD[5], B)
    ok["B2 rad_mix zero: the carried state['buf']['rad_fused'] = the radar block's output on the newest frame's own sweep (key frame and the next one)"] = \
        torch.equal(k1m["state"]["buf"]["rad_fused"], e2) and torch.equal(k2m["state"]["buf"]["rad_fused"], e5)
    memd, refd = twins(radar_kind="delta_slots", c_rad=64, **MOTION)       # the side loss's input and the motion block, with the memory arm
    d1m, d1r = run(memd, mk([0, 1, 2])), run(refd, mk([0, 1, 2]))
    side, want = d1m.get("radar_side"), d1r["radar_side"]
    with torch.no_grad():
        f, st = memd.radar(RAD[2], B, with_state=True)
    ok["B3 delta_slots + motion block: radar_side = the radar block's features and raster on radar_seq[0], = the arm without the memory, outputs bit for bit"] = \
        side is not None and torch.equal(side["features"], f) and torch.equal(side["state"], st) and torch.equal(side["features"], want["features"]) \
        and torch.equal(side["state"], want["state"]) and same(d1m, d1r)
    with torch.no_grad():
        mem.rad_mix.weight.normal_(0, 0.5)
    awake = run(mem, mk([0, 1, 2]))
    ok["B4 with rad_mix awake the carried radar buffer and the outputs differ (B1, B2 are not vacuous)"] = \
        float((awake["state"]["buf"]["rad_fused"] - e2).abs().max()) > 0.1 and gap(awake, k1r, state=False) > 1e-4
    return ok


def core_checks():
    """the equalities and rules the mutants must break: name -> bool (an exception counts as a failed group)"""
    ok = {}
    for fn in (stream_checks, grad_checks, mask_checks, identity_checks):
        try:
            ok.update(fn())
        except Exception as e:                                        # noqa: BLE001 - a planted bug that crashes is caught as well
            ok[f"{fn.__name__} raised {type(e).__name__}"] = False
    return ok


print("A - D: stream = batch, gradient, write mask and ego step per frame, step 0 = the network without the memory")
t0 = time.time()
for k, v in core_checks().items():
    check(v, k)

# ------------------------------------------------------------------------------------------------ E nothing else changes, refusals
print("E: default unchanged, refusals")
on, off = build(**MEM), build(**dict(MEM, radar_per_frame=False))
plain = mk([0, 1, 2])
no_seq = {k: v for k, v in plain.items() if k != "radar_seq"}
check(gap(run(off, plain), run(off, no_seq)) == 0, "radar_per_frame off: a batch with radar_seq gives the outputs of one without, bit for bit")
check(set(on.state_dict()) - set(off.state_dict()) == {"rad_mix.weight", "rad_mix.bias"} and not set(off.state_dict()) - set(on.state_dict()),
      "radar_per_frame adds exactly rad_mix.weight and rad_mix.bias")
refused = refuser(TINY)
check(refused(radar_per_frame=True) and refused(radar_per_frame=True, cam_buffer="recurrent") and refused(radar_per_frame=True, rad_temporal="window_memory")
      and refused(radar_per_frame=True, cam_buffer="pair", rad_temporal="window_memory") and refused(radar_per_frame=True, cam_buffer="recurrent", rad_temporal="ego_gru")
      and refused(radar_per_frame=True, radar_kind="off", **{k: v for k, v in MEM.items() if k != "radar_per_frame"}) and refused(radar_per_frame=1)
      and not refused(**MEM) and not refused(cam_buffer="recurrent", rad_temporal="window_memory"),
      "check_config: radar_per_frame needs cam_buffer 'recurrent', rad_temporal 'window_memory' and a radar, and a bool; the same network without it is fine")


def refuses(call):
    try:
        call()
        return False
    except ValueError:
        return True


check(refuses(lambda: run(on, no_seq)) and refuses(lambda: run(on, dict(plain, radar_seq=plain["radar_seq"][:2]))),
      "forward refuses a batch without radar_seq, and one with fewer sets than camera frames")
check(refuses(lambda: build(cam_buffer="recurrent", rad_temporal="window_memory").advance(mk([0]))) and not refuses(lambda: on.advance(mk([0]))),
      "advance refuses a radar memory without radar_per_frame (its state would go stale) and accepts it with")

# ------------------------------------------------------------------------------------------------ F plumbing
print("F: plumbing of batch['radar_seq']")
check(check_inputs(plain) == [] and any("radar_seq" in e for e in check_inputs(dict(plain, radar_seq=plain["radar_seq"][:2]))), "interface.check_inputs accepts radar_seq and counts its sets")
meta = T.to_device(plain, "meta")
check(len(meta["radar_seq"]) == 3 and all(t.device.type == "meta" for r in meta["radar_seq"] for t in r.values()) and meta["radar"]["x"].device.type == "meta",
      "train.to_device moves every set of radar_seq")
once = flip_batch(dict(plain), [True, False])
mirrored = [(a, b) for a, b in zip(plain["radar_seq"], once["radar_seq"])] + [(plain["radar"], once["radar"])]
check(all(torch.equal(b["y"][a["b"] == 0], -a["y"][a["b"] == 0]) and torch.equal(b["psi"][a["b"] == 0], -a["psi"][a["b"] == 0])
          and torch.equal(b["y"][a["b"] == 1], a["y"][a["b"] == 1]) and torch.equal(b["x"], a["x"]) for a, b in mirrored)
      and all(torch.equal(a["y"], RAD[2 - k]["y"]) for k, a in enumerate(plain["radar_seq"])),
      "flip_batch mirrors y and psi of every set of radar_seq for the flipped sample only (and does not touch the input)")
check(all(torch.equal(a["y"], b["y"]) for a, b in zip(flip_batch(once, [True, False])["radar_seq"], plain["radar_seq"])), "flipping twice is the identity")
pre = Precompute.from_model(on)
check(torch.equal(pre.radar_mask("radar_hits", RAD[1], B), radar_hits(RAD[1], B, on.grid, 1)) and torch.equal(pre.radar_mask("radar_static", RAD[1], B), radar_static(RAD[1], B, on.grid, 1.0, 1))
      and torch.equal(pre(dict(plain), ("radar_hits",))["radar_hits"], radar_hits(plain["radar"], B, on.grid, 1)),
      "Precompute.radar_mask = the maps of before (radar_hits, radar_static), and the batch path uses it")


def keyframe(batch):
    """the fields run_clip reads from a key frame: the ego steps to the neighbouring key frames, the tokens (the GT is not needed: the loss is stubbed below)"""
    gaps = {"prev": [(-1 / 12, 8.0, 0.05)] * 6, "next": [(1 / 12, 8.0, 0.05)] * 6}
    return dict(batch, gap_steps=[gaps] * B, tokens=["a", "b"])


def stub_forward_loss(model, loss, batch, step_frac, device, amp, cfg=None, diag=False, train_terms=True, state=None, step=None):
    res = model(batch, dts=(0.0,), state=state, step=step)
    return {"total": loss_of(res)}, res, None


m = build(**MEM).train()
seen, orig_frames, orig_loss = [], m._radar_frames, T.forward_loss
m._radar_frames = lambda batch, like: seen.append([r["y"].clone() for r in batch["radar_seq"]]) or orig_frames(batch, like)
clip = [keyframe(mk(idx)) for idx in ([0, 1, 2], [3, 4, 5])]
m.zero_grad()
T.forward_loss = stub_forward_loss                                    # (GtGenerator v13 needs a GridSpec, the model's Grid breaks synthetic_gt: the loss is not under test here)
try:
    _, finite, _, totals = T.run_clip(m, None, clip, 1.0, "cpu", "off", T.TrainConfig(seq_bptt=True, accum=1, flip_aug=1.0, gap_ego=True), backward=True, augment=True)
finally:
    T.forward_loss = orig_loss
check(finite and len(totals) == 2 and len(seen) == 2 and all(torch.equal(a, -RAD[2 - k]["y"]) for k, a in enumerate(seen[0]))
      and all(torch.equal(a, -RAD[5 - k]["y"]) for k, a in enumerate(seen[1])),
      "train.run_clip with flip_aug 1.0 hands the model the mirrored sets of radar_seq in both key frames (and the loss is finite)")
check(all(q.grad is not None and torch.isfinite(q.grad).all() for q in list(m.rad_mix.parameters()) + list(m.rad_temporal.parameters())),
      "the trainer's loop sends a gradient to rad_mix and the radar memory block")

# ------------------------------------------------------------------------------------------------ G mutants
print("G: planted bugs")


def mutants():
    recur, frames, camb = M.BEVFusionModel._recur, M.BEVFusionModel._radar_frames, M.BEVFusionModel._camera_buffered

    def no_radar_carry(block, bevs, moves, state, fused, obs, residual=None):
        return recur(block, bevs, moves, state, None if residual is not None else fused, obs, residual)

    def wrong_frames(self, batch, like):
        r = frames(self, batch, like)
        return dict(r, bevs=r["bevs"].flip(1))

    def wrong_masks(self, batch, like):
        r = frames(self, batch, like)
        return dict(r, obs=r["obs"][::-1])

    def no_radar_state(self, frames_, steps, radar, state, step, obs, rad_in=None):
        return camb(self, frames_, steps, radar, {k: v for k, v in state.items() if k != "rad"}, step, obs, rad_in)

    def no_side(self, batch, like):
        return dict(frames(self, batch, like), side=None)

    def side_from_oldest(self, batch, like):
        return dict(frames(self, batch, like), side=self._encode_radar(batch["radar_seq"][-1], like.shape[0], like.device)[1])

    def negate(i):                                                    # component i (1 = v, 2 = omega) of the radar's ego chains, the camera's stay
        def wrong_moves(block, bevs, moves, state, fused, obs, residual=None):
            if residual is not None:
                moves = [[tuple(-x if j == i else x for j, x in enumerate(s)) for s in mv] for mv in moves]
            return recur(block, bevs, moves, state, fused, obs, residual)
        return staticmethod(wrong_moves)

    orig = TemporalFusion.forward
    return {
        "the carried radar buffer is ignored (every call cold)": [(M.BEVFusionModel, "_recur", staticmethod(no_radar_carry))],
        "the radar frames are encoded in the wrong order": [(M.BEVFusionModel, "_radar_frames", wrong_frames)],
        "the write masks come in the wrong order": [(M.BEVFusionModel, "_radar_frames", wrong_masks)],
        "the radar state is not carried": [(M.BEVFusionModel, "_camera_buffered", no_radar_state)],
        "the radar side loss gets no features": [(M.BEVFusionModel, "_radar_frames", no_side)],
        "the radar side loss reads the oldest frame": [(M.BEVFusionModel, "_radar_frames", side_from_oldest)],
        "the radar's ego chains have the wrong sign of v": [(M.BEVFusionModel, "_recur", negate(1))],
        "the radar's ego chains have the wrong sign of omega": [(M.BEVFusionModel, "_recur", negate(2))],
        "the carried fused state is cut by a detach": [(TemporalFusion, "forward", lambda self, cur, past=(), *a, **k: orig(self, cur, [(x.detach(), s) for x, s in past], *a, **k))],
    }


for name, patches in mutants().items():
    saved = [(o, a, vars(o)[a]) for o, a, _ in patches]              # vars(): a staticmethod stays one
    for o, a, f in patches:
        setattr(o, a, f)
    try:
        try:
            died = [k for k, v in core_checks().items() if not v]
        except Exception as e:                                       # noqa: BLE001 - a mutant that crashes a check is caught as well
            died = [f"raised {type(e).__name__}"]
    finally:
        for o, a, f in saved:
            setattr(o, a, f)
    check(bool(died), f"mutant killed: {name} -> {died[0] if died else 'SURVIVED'}{' (+%d)' % (len(died) - 1) if len(died) > 1 else ''}")
check(all(core_checks().values()), "after the mutants the baseline passes again")

# ------------------------------------------------------------------------------------------------ H configs
print("H: configs/t25_*.json")
CFG = os.path.join(ROOT, "training", "configs")
load = lambda n: json.load(open(os.path.join(CFG, n + ".json")))      # noqa: E731
p4, rec = load("t21_pair4"), load("t21_rec")
WANT = {"t25_ctl": (p4, {"batch_size": 2, "accum": 3}, {}),
        "t25_rec_r": (rec, {}, {}),
        "t25_rec_scan": (rec, {"radar_window": 0.09}, {}),
        "t25_rec_scan_mem": (rec, {"radar_window": 0.09}, {"radar_per_frame": True, "rad_temporal": "window_memory", "rad_static_gate": True})}
for name, (src, top, model) in WANT.items():
    got = load(name)
    diff = {k for k in set(src) | set(got) if src.get(k) != got.get(k)}
    mdiff = {k for k in set(src["model"]) | set(got["model"]) if src["model"].get(k) != got["model"].get(k)}
    want_top = {"_about", "name"} | set(top) | ({"model"} if model else set())
    check(got["name"] == name and diff == want_top and all(got[k] == v for k, v in top.items()) and mdiff == set(model) and all(got["model"][k] == v for k, v in model.items()),
          f"{name}: differs from {'t21_pair4' if src is p4 else 't21_rec'} in {sorted(want_top - {'_about', 'name'})} and {sorted(model)} only")
    cfg = T.TrainConfig(**{k: v for k, v in got.items() if not k.startswith("_") and k != "split"})      # ("split" is resolved to scene lists by training/splits.py before the trainer reads it)
    check(not check_config(ModelConfig(**cfg.model)) and cfg.steps == 1500 and cfg.seed == 0 and cfg.init_from == "runs/t20_base_s0/ckpt_00006000.pt" and cfg.batch_size * cfg.accum == 6,
          f"{name}: its keys are TrainConfig fields, passes check_config, 1500 steps, seed 0, warm start t20_base_s0, 6 clips per step")

print(f"\n{'FAILED: ' + str(len(fails)) + ' check(s)' if fails else 'test_radar_recursive passed'} ({time.time() - t0:.0f} s)")
sys.exit(1 if fails else 0)
