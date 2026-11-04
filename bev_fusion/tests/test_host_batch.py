"""The host copies of the batch (D54; training/train.py host_side / batch_ego / clip_step, lift_pd.lift_key, launch_saving.to_dev / index): the same numbers with fewer waits for the GPU.
CPU, no data, about a minute; the parts that need a GPU run when one is there (a few MB, the sync debug mode of torch.cuda). Nothing may change a value:

  1. batch_ego / clip_step equal the old per-tensor versions (values, dtype, shape, order of the dts, the strict error), and their tensors are views of ONE allocation
     (the old ones were separate tensors, one copy to the GPU each)
  2. host_side takes references to the CPU tensors K, T_cam_car, ida, ida2, flip - nothing else, nothing that is already on the GPU - and forward_loss puts them into the batch
  3. the projection-table cache key made from the host copies is the key made from the batch tensors; the lift reads no GPU tensor for it when it is given the key, the cached
     table is reused and a different calibration still builds a new table
  4. camera_visibility with the host copies gives the mask of the plain call (heights as a tuple or a tensor, oversampled), reads the model buffers once, and a buffer changed in place
     is noticed
  5. the model with sync_free and a batch with batch["host"] gives every output, loss term and gradient bit for bit as the same model without the copies, and without sync_free the
     copies are not read at all; 'any sample flipped?' is answered from the host copy when there is one
  6. launch_saving.to_dev / index and timed_head.dt_embedding give the plain values (and are the plain calls with the switch off)
  7. with a GPU: the ego tensors, the pinned copies, the cached index and the keyed lift do not wait for the GPU (torch.cuda.set_sync_debug_mode("error")); the old forms do - which is
     the proof that this part of the test can fail
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_host_batch.py   (PYTHONPATH = the repository root for autogt)"""


import math
import os
import random
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), os.path.join(ROOT, "..")]
import launch_saving as LS  # noqa: E402
import lift_pd as LP  # noqa: E402
import losses as L  # noqa: E402
import model as M  # noqa: E402
import precompute as PC  # noqa: E402
import synthetic_world as sw  # noqa: E402
import timed_head as TH  # noqa: E402
import train as T  # noqa: E402
from feature_cache import CachedBEVFusionModel  # noqa: E402
from lift_pd import Grid  # noqa: E402
from model import ModelConfig  # noqa: E402
from test_slots import BASE, batch_for  # noqa: E402

fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


# ---- 1 ego tensors
def old_batch_ego(gap_steps, dts, device, strict=False):          # the function as it was before D54
    ego = {}
    for dt in dts:
        if abs(dt) < 1e-6:
            continue
        if strict and abs(abs(dt) - 0.5) > 1e-6:
            raise ValueError(f"batch_ego: dt {dt:+.2f} s has no ego gap")
        key = "prev" if dt < 0 else "next"
        vals = [T.gap_ego_motion(g[key]) for g in gap_steps]
        ego[dt] = tuple(torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(3))
    return ego


def old_clip_step(prev_batch, device):
    vals = [T.gap_ego_motion(g["next"]) for g in prev_batch["gap_steps"]]
    return tuple(torch.tensor([v[i] for v in vals], dtype=torch.float32, device=device) for i in range(3))


rng = random.Random(3)


def gap_steps(B):
    def steps():
        return [(rng.uniform(0.04, 0.2), rng.uniform(0, 20), rng.uniform(-0.3, 0.3)) for _ in range(rng.choice([1, 2, 3, 6]))]
    return [{"prev": [(-dt, v, w) for dt, v, w in steps()], "next": steps()} for _ in range(B)]


same = True
for B in (1, 3, 5):
    gs = gap_steps(B)
    for dts in ((-0.5, 0.0, 0.5), (0.5,), (0.0,), (0.5, -0.5), ()):
        a, b = old_batch_ego(gs, dts, "cpu"), T.batch_ego(gs, dts, "cpu")
        same &= list(a) == list(b) and all(len(a[k]) == len(b[k]) == 3 and all(torch.equal(x, y) and x.dtype == y.dtype == torch.float32 and x.shape == y.shape == (B,)
                                                                                for x, y in zip(a[k], b[k])) for k in a)
    c, d = old_clip_step({"gap_steps": gs}, "cpu"), T.clip_step({"gap_steps": gs}, "cpu")
    same &= len(d) == 3 and all(torch.equal(x, y) and x.dtype == y.dtype and x.shape == y.shape for x, y in zip(c, d))
check(same, "batch_ego and clip_step give the old values (B = 1, 3, 5; dts with and without 0; none)")
gs = gap_steps(3)
e = T.batch_ego(gs, (-0.5, 0.0, 0.5), "cpu")
ptrs = {t.untyped_storage().data_ptr() for k in e for t in e[k]}
check(len(ptrs) == 1 and len({t.untyped_storage().data_ptr() for t in old_batch_ego(gs, (-0.5, 0.5), "cpu")[0.5]}) == 3, "the ego tensors of all dts are views of one allocation (the old ones: one each)")
check(len({t.untyped_storage().data_ptr() for t in T.clip_step({"gap_steps": gs}, "cpu")}) == 1, "clip_step: one allocation")
for fn, name in ((old_batch_ego, "old"), (T.batch_ego, "new")):
    try:
        fn(gs, (0.1,), "cpu", strict=True)
        check(False, f"batch_ego ({name}) strict: dt 0.1 must raise")
    except ValueError:
        check(True, f"batch_ego ({name}) strict: dt 0.1 raises")
check(T.batch_ego(gs, (-0.5, 0.5), torch.device("cpu"))[0.5][0].shape == (3,), "batch_ego takes a torch.device as well")

# ---- 2 host_side
K = torch.randn(2, 3, 3, 3)
b = {"K": K, "T_cam_car": torch.randn(2, 3, 4, 4), "ida": torch.randn(3, 3), "flip": torch.tensor([True, False]), "v": torch.zeros(2), "hw": (448, 800), "gt": [{}, {}]}
h = T.host_side(b)
check(set(h) == {"K", "T_cam_car", "ida", "flip"} and h["K"] is K, "host_side: references to K, T_cam_car, ida, flip (not v, hw, gt); no copy")
check(T.host_side({"K": K, "ida": None, "hw": (1, 2)}) == {"K": K}, "host_side: a missing or non-tensor entry is skipped")
check(set(T.host_side(dict(b, ida2=torch.eye(3)))) == {"K", "T_cam_car", "ida", "ida2", "flip"}, "host_side: ida2 (the BEVFusion crop) too")
if torch.cuda.is_available():
    check(set(T.host_side(dict(b, K=K.cuda()))) == {"T_cam_car", "ida", "flip"}, "host_side: a tensor that is on the GPU already is not a host copy")

# ---- 3 the projection-table cache key
dc = torch.linspace(1.0, 60.0, 119)
Kc, Tc, hw = LP._front_camera(torch.float32)
ida = torch.tensor([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1.0]])
grid = Grid()
check(LP.lift_key(Kc, Tc, hw, ida) == (tuple(Kc.flatten().tolist()), tuple(Tc.flatten().tolist()), tuple(hw), tuple(ida.flatten().tolist())) and LP.lift_key(Kc, Tc, hw)[3] is None,
      "lift_key: intrinsics, extrinsics, image size, crop transform (None without one)")
lift = LP.LiftPD(grid, dc)
feat = torch.randn(2, 8, 12, 20)
prob = torch.softmax(torch.randn(2, len(dc), 12, 20), 1)
calls = []
_lk = LP.lift_key
LP.lift_key = lambda *a, **k: (calls.append(1), _lk(*a, **k))[1]
o_plain = lift(feat, prob, Kc, Tc, hw, ida)
n_plain = len(calls)
k_host = _lk(Kc.clone(), Tc.clone(), hw, ida.clone())
o_keyed = lift(feat, prob, Kc, Tc, hw, ida, key=k_host)
check(n_plain == 1 and len(calls) == 1 and torch.equal(o_plain, o_keyed) and lift._cache[0] == k_host, "the lift: a given key is used as it is (no key made inside), the output is the plain one, the cache holds it")
K2 = Kc.clone()
K2[0, 0] += 3.0
o2 = lift(feat, prob, K2, Tc, hw, ida, key=_lk(K2, Tc, hw, ida))
o2_plain = LP.LiftPD(grid, dc)(feat, prob, K2, Tc, hw, ida)
check(torch.equal(o2, o2_plain) and not torch.equal(o2, o_keyed) and lift._cache[0] == _lk(K2, Tc, hw, ida), "a different calibration builds a new table (the key decides), equal to the plain lift")
LP.lift_key = _lk

# ---- 4 camera_visibility
z = tuple(grid.z_levels)
dcb = torch.linspace(1.0, 60.0, 119)
for zl, over in ((z, 1), (torch.tensor(z), 1), (torch.tensor(z), 2)):
    LP._VIS_CACHE.clear()
    LP._HOST_CONST.clear()
    a = LP.camera_visibility(grid, zl, dcb, Kc, Tc, hw, ida, over)
    LP._VIS_CACHE.clear()
    bm = LP.camera_visibility(grid, zl, dcb, Kc, Tc, hw, ida, over, host=(Kc.clone(), Tc.clone(), ida.clone()))
    again = LP.camera_visibility(grid, zl, dcb, Kc, Tc, hw, ida, over, host=(Kc.clone(), Tc.clone(), ida.clone()))
    check(torch.equal(a, bm) and bm is again and len(LP._HOST_CONST) == 1, f"camera_visibility with the host copies = the plain mask (heights {type(zl).__name__}, oversample {over}); a second call hits the cache and the host constants")
zt = torch.tensor(z)
LP._VIS_CACHE.clear()
LP._HOST_CONST.clear()
m0 =LP.camera_visibility(grid, zt, dcb, Kc, Tc, hw, ida, 1, host=(Kc, Tc, ida))
zt[0] += 0.5                                                 # a buffer changed in place: the host constants must be read again
LP._VIS_CACHE.clear()
m1 = LP.camera_visibility(grid, zt, dcb, Kc, Tc, hw, ida, 1, host=(Kc, Tc, ida))
LP._VIS_CACHE.clear()
check(torch.equal(m1, LP.camera_visibility(grid, zt, dcb, Kc, Tc, hw, ida, 1)) and len(LP._HOST_CONST) == 2, "a height buffer changed in place is noticed (the host constants follow, the mask equals the plain one)")

# ---- 5 the model
B_ = sw.Box
Bn = 2


def with_gt(batch_, grid_):
    scene = sw.Scene([B_(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), B_(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                      B_(13.0, -4.0, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0), B_(5.0, 5.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)],
                     lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": L.synthetic_gt(scene.step(-0.5), grid_), "cur": L.synthetic_gt(scene, grid_), "next": L.synthetic_gt(scene.step(0.5), grid_)}
    return dict(batch_, gt=[side] * batch_["v"].shape[0])


cfg = ModelConfig(**dict(BASE, camera_features="dual", cam_temporal="window_memory", sync_free=True))
torch.manual_seed(0)
m = CachedBEVFusionModel(cfg).eval()                         # eval: no modality dropout, the same random numbers in every run
with torch.no_grad():
    for q in m.parameters():
        if q.requires_grad and q.abs().max() == 0:
            q.normal_(0, 1e-2)
batch = with_gt(batch_for(cfg, B=Bn), m.grid)
batch["gap_steps"] = gap_steps(Bn)
loss = L.BEVLoss(grid=m.grid)
_hs = T.host_side
gk, vk = [], []                                              # which keys the lift made inside, which hosts the visibility got
_lk = LP.lift_key
_cv = PC.camera_visibility
LP.lift_key = lambda *a, **k: (gk.append(1), _lk(*a, **k))[1]
PC.camera_visibility = lambda *a, **k: (vk.append(k.get("host") is not None), _cv(*a, **k))[1]


def run(with_host, flags_on=True, flip=True):
    b = dict(batch)
    b["flip"] = torch.tensor([True, False]) if flip else torch.zeros(Bn, dtype=torch.bool)
    T.host_side = _hs if with_host else (lambda bb: {})
    m.cfg.sync_free = loss.cfg.sync_free = flags_on
    for mod in m.modules():
        if isinstance(mod, LP.LiftPD):
            mod._cache = None
    LP._VIS_CACHE.clear()
    gk.clear()
    vk.clear()
    m.zero_grad()
    torch.manual_seed(0)
    out, res, _ = T.forward_loss(m, loss, b, 0.5, "cpu", "off", None)
    out["total"].backward()
    grads = {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}
    T.host_side = _hs
    return out, res, grads, len(gk), list(vk)


def flat(res):
    d = {}
    for dt, p in res["out"].items():
        for k, v in p.items():
            for kk, t in (v.items() if isinstance(v, dict) else [("", v)]):
                if torch.is_tensor(t):
                    d[(dt, k, kk)] = t
    return d


def same_run(a, b):
    fa, fb = flat(a[1]), flat(b[1])
    outs = set(fa) == set(fb) and all(torch.equal(fa[k], fb[k]) for k in fa)
    terms = set(a[0]) == set(b[0]) and all(torch.equal(torch.as_tensor(a[0][k]), torch.as_tensor(b[0][k])) for k in a[0])
    grads = set(a[2]) == set(b[2]) and all(torch.equal(a[2][k], b[2][k]) for k in a[2])
    return outs, terms, grads, len(fa), len(a[0]), len(a[2])


try:
    ref = run(with_host=False)
    got = run(with_host=True)
    off = run(with_host=True, flags_on=False)
    off_ref = run(with_host=False, flags_on=False)
    nf = run(with_host=True, flip=False)
    nf_ref = run(with_host=False, flip=False)
finally:
    LP.lift_key = _lk
    PC.camera_visibility = _cv
    T.host_side = _hs
o, t_, g_, n1, n2, n3 = same_run(ref, got)
check(o and t_ and g_, f"sync_free with batch['host'] (one sample flipped): {n1} output tensors, {n2} loss terms and {n3} gradients equal the run without the copies bit for bit")
check(ref[3] > 0 and got[3] == 0, f"the lifts made {ref[3]} keys of their own without the copies and {got[3]} with them (the keys came from the host copies)")
check(got[4] and all(got[4]) and not any(off[4]), "the visibility got the host copies with sync_free and none without")
o, t_, g_, *_ = same_run(off_ref, off)
check(o and t_ and g_ and off[3] == off_ref[3] > 0, "without sync_free the copies are not read: the lifts make their keys, everything equals the run without the copies")
o, t_, g_, *_ = same_run(nf_ref, nf)
check(o and t_ and g_, "no sample flipped: equal as well")
o, t_, g_, *_ = same_run(ref, off_ref)
check(o and t_, "sync_free on / off: outputs and loss terms equal (the gradients to rounding is the job of test_launch_saving)")
fl_dev = {"flip": torch.zeros(2, dtype=torch.bool)}
check(M.BEVFusionModel._flip_any(fl_dev) is False and M.BEVFusionModel._flip_any(dict(fl_dev, host={"flip": torch.tensor([True, False])})) is True
      and M.BEVFusionModel._flip_any({}) is False and M.BEVFusionModel._flip_any({"flip": torch.tensor([False, True])}) is True, "_flip_any: from the host copy when there is one, else from the tensor, False without flags")
check(M.BEVFusionModel._lift_keys({"K": 1}, hw) is None and M.BEVFusionModel._lift_keys({"ida": ida, "host": {"K": torch.zeros(2, 1, 3, 3), "T_cam_car": torch.zeros(2, 1, 4, 4)}}, hw) is None,
      "_lift_keys: no key without the copies, none when the copy of the crop transform is missing")
seen = {}
_fwd = m.forward
m.forward = lambda b, *a, **k: (seen.update(host=b.get("host")), _fwd(b, *a, **k))[1]
T.forward_loss(m, loss, dict(batch, flip=torch.tensor([True, False])), 0.5, "cpu", "off", None)
m.forward = _fwd
check(seen.get("host") is not None and seen["host"]["K"] is batch["K"] and seen["host"]["flip"].tolist() == [True, False], "forward_loss: batch['host'] holds the CPU tensors of the batch")

# ---- 6 to_dev, index, dt_embedding
t_ = torch.randn(3, 4)
check(LS.to_dev(t_, "cpu") is t_ and LS.index([2, 0], "cpu") == [2, 0], "switch off: to_dev is .to(device), index is the list")
with LS.enabled(sync=True):
    ix = LS.index([2, 0], "cpu")
    check(torch.is_tensor(ix) and ix.tolist() == [2, 0] and LS.index([2, 0], "cpu") is ix and torch.equal(t_[:, ix], t_[:, [2, 0]]), "switch on: the index tensor is made once and indexes like the list")
    check(torch.equal(LS.to_dev(t_, "cpu"), t_), "switch on, CPU: to_dev gives the tensor")
ok = True
for like in (torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.float16)):
    for dt in (0.5, -0.5, 0.0, 1, 0.1, 1 / 3):
        a_ = TH.dt_embedding(dt, 4, like)
        with LS.enabled(sync=True):
            b_ = TH.dt_embedding(dt, 4, like)
        ok &= torch.equal(a_, b_) and a_.dtype == b_.dtype and a_.shape == b_.shape == (4, 3)
check(ok, "dt_embedding with the switch on (a number filled on the device) equals the tensor form: float32 / float64 / bfloat16 / float16, dt 0.5, -0.5, 0, 1, 0.1, 1/3")
t_dev = torch.tensor([0.5, -0.5])
check(torch.equal(TH.dt_embedding(t_dev, 2, torch.zeros(2)), torch.stack([t_dev, t_dev.abs(), torch.sign(t_dev)], 1)), "dt_embedding: a (B,) tensor is as before")

# ---- 7 no waits on the GPU
if torch.cuda.is_available():
    dev = "cuda"
    torch.zeros(1, device=dev)
    gs = gap_steps(3)

    def waits(fn):
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            fn()
            return False
        except RuntimeError:
            return True
        finally:
            torch.cuda.set_sync_debug_mode("default")
    big = torch.randn(4096, 4096, device=dev)
    for _ in range(3):
        big = big @ big * 1e-3                               # a queue of work, so a pageable copy has something to wait for
    check(waits(lambda: old_batch_ego(gs, (-0.5, 0.5), dev)), "(detector) the old ego tensors wait for the GPU (the sync debug mode raises)")
    check(waits(lambda: old_clip_step({"gap_steps": gs}, dev)), "(detector) the old clip step waits for the GPU")
    check(not waits(lambda: T.batch_ego(gs, (-0.5, 0.0, 0.5), dev)), "batch_ego: no wait for the GPU")
    check(not waits(lambda: T.clip_step({"gap_steps": gs}, dev)), "clip_step: no wait for the GPU")
    e = T.batch_ego(gs, (-0.5, 0.5), dev)
    ref_e = old_batch_ego(gs, (-0.5, 0.5), dev)
    torch.cuda.synchronize()
    check(all(torch.equal(e[k][i], ref_e[k][i]) and e[k][i].device.type == "cuda" for k in e for i in range(3)), "on the GPU: the ego tensors equal the old ones")
    with LS.enabled(sync=True):
        LS.index([1, 2], dev)                                # made once (this is the one pageable copy)
        check(not waits(lambda: LS.index([1, 2], dev)), "index: a second call makes no copy")
        tt = torch.randn(7, 5)
        check(not waits(lambda: LS.to_dev(tt, dev)), "to_dev: a CPU tensor goes from pinned memory without a wait")
        check(waits(lambda: tt.to(dev)), "(detector) the plain .to(device) of a pageable tensor waits")
        check(torch.equal(LS.to_dev(tt, dev).cpu(), tt), "to_dev: the values arrive")
        check(not waits(lambda: TH.dt_embedding(0.5, 3, torch.zeros(3, device=dev))), "dt_embedding: a number needs no copy")
    check(waits(lambda: TH.dt_embedding(0.5, 3, torch.zeros(3, device=dev))), "(detector) dt_embedding without the switch copies the number")
    lift_g = LP.LiftPD(grid, dc.to(dev)).to(dev)
    Kg, Tg, idag = Kc.to(dev), Tc.to(dev), ida.to(dev)
    f_g, p_g = feat.to(dev), prob.to(dev)
    key = LP.lift_key(Kc, Tc, hw, ida)
    lift_g(f_g, p_g, Kg, Tg, hw, idag, key=key)             # builds the table
    check(not waits(lambda: lift_g(f_g, p_g, Kg, Tg, hw, idag, key=key)), "the keyed lift with a cached table: no wait for the GPU")
    check(waits(lambda: lift_g(f_g, p_g, Kg, Tg, hw, idag)), "(detector) the lift that makes its key from the GPU tensors waits")
    LP._VIS_CACHE.clear()
    LP._HOST_CONST.clear()
    zl, dcg = torch.tensor(z, device=dev), dc.to(dev)
    LP.camera_visibility(grid, zl, dcg, Kg, Tg, hw, idag, 1, host=(Kc, Tc, ida))
    check(not waits(lambda: LP.camera_visibility(grid, zl, dcg, Kg, Tg, hw, idag, 1, host=(Kc, Tc, ida))), "camera_visibility with the host copies and a cached mask: no wait for the GPU")
    check(waits(lambda: LP.camera_visibility(grid, zl, dcg, Kg, Tg, hw, idag, 1)), "(detector) camera_visibility without them waits")
else:
    print("  skip  part 7 (no GPU)")

print("HOST BATCH GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)}): " + "; ".join(fails[:5])))
sys.exit(1 if fails else 0)
