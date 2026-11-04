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


import dataclasses
import math
import os
import random
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [os.path.join(ROOT, "training"), os.path.join(ROOT, "tests"), os.path.join(ROOT, "tests", "support"), os.path.join(ROOT, "..")]
import launch_saving as LS  # noqa: E402
import lift_pd as LP  # noqa: E402
import losses as L  # noqa: E402
import object_head as OH  # noqa: E402
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
m0 = LP.camera_visibility(grid, zt, dcb, Kc, Tc, hw, ida, 1, host=(Kc, Tc, ida))
zt[0] += 0.5                                                 # a buffer changed in place: the host constants must be read again
LP._VIS_CACHE.clear()
m1 = LP.camera_visibility(grid, zt, dcb, Kc, Tc, hw, ida, 1, host=(Kc, Tc, ida))
LP._VIS_CACHE.clear()
check(torch.equal(m1, LP.camera_visibility(grid, zt, dcb, Kc, Tc, hw, ida, 1)) and len(LP._HOST_CONST) == 2, "a height buffer changed in place is noticed (the host constants follow, the mask equals the plain one)")
# a cache hit must never serve another calibration: K, T_cam_car, the crop transform and the image size each decide the key (the calls follow one another, no cache clear between them)
K_b = Kc.clone()
K_b[0, 0], K_b[1, 1] = 300.0, 300.0
T_b = LP._front_camera(torch.float32, x0=12.0)[1]
ida_b = torch.eye(3)
base = dict(K=Kc, T=Tc, ida=ida, hw=hw)
variants = {"base": base, "K": dict(base, K=K_b), "T_cam_car": dict(base, T=T_b), "ida": dict(base, ida=ida_b), "image size": dict(base, hw=(224, 400))}
plain_m, host_m = {}, {}
for name, v in variants.items():
    LP._VIS_CACHE.clear()
    plain_m[name] = LP.camera_visibility(grid, z, dcb, v["K"], v["T"], v["hw"], v["ida"], 1)
LP._VIS_CACHE.clear()
LP._HOST_CONST.clear()
for name, v in variants.items():
    host_m[name] = LP.camera_visibility(grid, z, dcb, v["K"], v["T"], v["hw"], v["ida"], 1, host=(v["K"].clone(), v["T"].clone(), v["ida"].clone()))
check(all(not torch.equal(plain_m[n], plain_m["base"]) for n in variants if n != "base"), "(sensitivity) each of K, T_cam_car, ida, image size changes the visibility mask of this test")
check(all(torch.equal(host_m[n], plain_m[n]) for n in variants), "camera_visibility with the host copies, one call after the other: every calibration gets its own mask (the key holds K, T_cam_car, ida and the image size)")

# ---- 5 the model
B_ = sw.Box
Bn = 2


def with_gt(batch_, grid_):
    scene = sw.Scene([B_(6.0, -1.75, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0), B_(10.0, 4.0, 4.5, 1.8, 1.5, 0.0, 0.0, 0.0, "car", 10.0),
                      B_(13.0, -4.0, 8.0, 2.5, 3.0, 0.05, 0.0, 0.0, "truck", 20.0), B_(5.0, 5.5, 0.6, 0.6, 1.8, math.pi / 2, 0.0, 1.2, "pedestrian", -5.0)],
                     lanes=[-5.25, -1.75, 1.75, 5.25], v=8.0, omega=0.05)
    side = {"prev": L.synthetic_gt(scene.step(-0.5), grid_), "cur": L.synthetic_gt(scene, grid_), "next": L.synthetic_gt(scene.step(0.5), grid_)}
    return dict(batch_, gt=[side] * batch_["v"].shape[0])


def distinct_calibrations(b):
    """a different camera for every sample AND every frame (batch_for repeats one): a key made from the wrong sample or the wrong frame is then a wrong key"""
    b["K"][1, :, :2] *= 0.8
    b["T_cam_car"][1, :, 0, 3] += 0.3
    b["K"][:, 1:, 0, 0] += 5.0
    b["T_cam_car"][:, 1:, 1, 3] += 0.2
    return b


_lk = LP.lift_key                                            # the real function; the spies below wrap the names the code calls
_cv = PC.camera_visibility
_lf = LP.LiftPD.forward
_hs = T.host_side
SEEN = {"key_made_inside": 0, "lift": [], "vis": []}


def spy_lift(self, feat, depth_prob, K, T_cam_car, image_hw, ida=None, key=None):
    SEEN["lift"].append((key, _lk(K, T_cam_car, image_hw, ida)))      # the key it was given, and the key of the arguments it really got
    return _lf(self, feat, depth_prob, K, T_cam_car, image_hw, ida, key=key)


def spy_vis(grid_, z_levels, depth_centres, K, T_cam_car, image_hw, ida=None, oversample=1, slice_height=0.5, host=None):
    SEEN["vis"].append(None if host is None else (torch.equal(host[0], K) and torch.equal(host[1], T_cam_car)
                                                  and ((host[2] is None and ida is None) or (host[2] is not None and ida is not None and torch.equal(host[2], ida)))))
    return _cv(grid_, z_levels, depth_centres, K, T_cam_car, image_hw, ida, oversample, slice_height, host=host)


def part5(m, batch, loss, tag):
    def run(with_host, flags_on=True, flip=True):
        b = dict(batch)
        b["flip"] = torch.tensor([True, False]) if flip else torch.zeros(Bn, dtype=torch.bool)
        T.host_side = _hs if with_host else (lambda bb: {})
        m.cfg.sync_free = loss.cfg.sync_free = flags_on
        for mod in m.modules():
            if isinstance(mod, LP.LiftPD):
                mod._cache = None
        LP._VIS_CACHE.clear()
        SEEN.update(key_made_inside=0, lift=[], vis=[])
        m.zero_grad()
        torch.manual_seed(0)
        out, res, _ = T.forward_loss(m, loss, b, 0.5, "cpu", "off", None)
        out["total"].backward()
        grads = {n: q.grad.clone() for n, q in m.named_parameters() if q.grad is not None}
        T.host_side = _hs
        return out, res, grads, dict(SEEN)

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

    LP.LiftPD.forward = spy_lift
    PC.camera_visibility = spy_vis
    try:
        ref = run(with_host=False)
        got = run(with_host=True)
        off = run(with_host=True, flags_on=False)
        off_ref = run(with_host=False, flags_on=False)
        nf = run(with_host=True, flip=False)
        nf_ref = run(with_host=False, flip=False)
    finally:
        LP.LiftPD.forward = _lf
        PC.camera_visibility = _cv
        T.host_side = _hs
    o, t_, g_, n1, n2, n3 = same_run(ref, got)
    check(o and t_ and g_, f"{tag}: sync_free with batch['host'] (one sample flipped, a different camera per sample and frame): {n1} output tensors, {n2} loss terms and {n3} gradients equal the run without the copies bit for bit")
    nl_ref, nl_got = len(ref[3]["lift"]), len(got[3]["lift"])
    check(nl_ref > 0 and nl_got == nl_ref and all(k is None for k, _ in ref[3]["lift"]), f"{tag}: without the copies the {nl_ref} lift calls get no key (they make their own)")
    check(all(k is not None and k == e for k, e in got[3]["lift"]) and nl_got >= 2 * Bn, f"{tag}: with the copies every one of the {nl_got} lift calls gets exactly the key of the K, T_cam_car, image size and crop it is called with "
                                                                                      "(both sources, every sample; the lane branch too when there is one)")
    check(len({e for _, e in got[3]["lift"]}) >= Bn, f"{tag}: the calls really differ in calibration ({len({e for _, e in got[3]['lift']})} distinct keys)")
    check(got[3]["vis"] and all(v is True for v in got[3]["vis"]) and all(v is None for v in off[3]["vis"]), f"{tag}: the visibility got host copies equal to its K, T_cam_car, ida with sync_free ({len(got[3]['vis'])} calls) and none without")
    o, t_, g_, *_ = same_run(off_ref, off)
    check(o and t_ and g_ and all(k is None for k, _ in off[3]["lift"]), f"{tag}: without sync_free the copies are not read (no lift got a key); everything equals the run without the copies")
    o, t_, g_, *_ = same_run(nf_ref, nf)
    check(o and t_ and g_, f"{tag}: no sample flipped: equal as well")
    o, t_, g_, *_ = same_run(ref, off_ref)
    check(o and t_, f"{tag}: sync_free on / off: outputs and loss terms equal (the gradients to rounding is the job of test_launch_saving)")


cfg = ModelConfig(**dict(BASE, camera_features="dual", cam_temporal="window_memory", sync_free=True))
for tag, extra in (("dual", {}), ("dual + lane branch", dict(lane_branch="res4", lane_pretrained=""))):
    torch.manual_seed(0)
    m = CachedBEVFusionModel(dataclasses.replace(cfg, **extra)).eval()      # eval: no modality dropout, the same random numbers in every run
    with torch.no_grad():
        for q in m.parameters():
            if q.requires_grad and q.abs().max() == 0:
                q.normal_(0, 1e-2)
    batch = distinct_calibrations(with_gt(batch_for(m.cfg, B=Bn), m.grid))
    batch["gap_steps"] = gap_steps(Bn)
    if extra:
        batch["lane_img"] = torch.randint(0, 256, (Bn, 3, *batch["hw"]), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
    part5(m, batch, L.BEVLoss(grid=m.grid), tag)
fl_dev = {"flip": torch.zeros(2, dtype=torch.bool)}
check(M.BEVFusionModel._flip_any(fl_dev) is False and M.BEVFusionModel._flip_any(dict(fl_dev, host={"flip": torch.tensor([True, False])})) is True
      and M.BEVFusionModel._flip_any({}) is False and M.BEVFusionModel._flip_any({"flip": torch.tensor([False, True])}) is True, "_flip_any: from the host copy when there is one, else from the tensor, False without flags")
check(M.BEVFusionModel._lift_keys({"K": 1}, hw) is None and M.BEVFusionModel._lift_keys({"ida": ida, "host": {"K": torch.zeros(2, 1, 3, 3), "T_cam_car": torch.zeros(2, 1, 4, 4)}}, hw) is None,
      "_lift_keys: no key without the copies, none when the copy of the crop transform is missing")
hb = {"K": batch["K"], "T_cam_car": batch["T_cam_car"], "ida": batch["ida"], "ida2": batch["ida2"]}
ks, k2 = M.BEVFusionModel._lift_keys({"ida": batch["ida"], "ida2": batch["ida2"], "host": hb}, batch["hw"], "ida"), M.BEVFusionModel._lift_keys({"ida": batch["ida"], "ida2": batch["ida2"], "host": hb}, batch["hw"], "ida2")
check(all(ks[b] == _lk(batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida"]) and k2[b] == _lk(batch["K"][b, 0], batch["T_cam_car"][b, 0], batch["hw"], batch["ida2"]) for b in range(Bn))
      and ks[0] != ks[1] and ks[0] != k2[0] and not torch.equal(batch["ida"], batch["ida2"]), "_lift_keys: sample b gets the key of K[b, 0] and T_cam_car[b, 0], frame 0; 'ida' and 'ida2' give different keys")
pre = PC.Precompute.from_model(m)                            # the visibility step: host copies are used only when they describe the same K, T_cam_car AND crop transform
SEEN["vis"] = []
PC.camera_visibility = spy_vis
try:
    pre.camera(dict(batch, host={"K": batch["K"], "T_cam_car": batch["T_cam_car"]}))                                  # the batch has an ida, the copies have none: not usable
    n_unusable = list(SEEN["vis"])
    pre.camera(dict(batch, host={"K": batch["K"], "T_cam_car": batch["T_cam_car"], "ida": batch["ida"]}))
finally:
    PC.camera_visibility = _cv
check(n_unusable == [None] * Bn and SEEN["vis"][Bn:] == [True] * Bn, "Precompute.camera: copies without the crop transform of a batch that has one are not used; complete copies are, each sample with its own K, T_cam_car")
seen = {}
_fwd = m.forward
m.forward = lambda b, *a, **k: (seen.update(host=b.get("host")), _fwd(b, *a, **k))[1]
T.forward_loss(m, L.BEVLoss(grid=m.grid), dict(batch, flip=torch.tensor([True, False])), 0.5, "cpu", "off", None)
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
        except RuntimeError as err:
            if "synchroniz" not in str(err):                 # only the sync debug mode's error counts as a wait, not any other failure
                raise
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
        td = LS.to_dev(tt, dev)
        check(td.device.type == "cuda" and torch.equal(td.cpu(), tt), "to_dev: the values arrive, on the GPU")
        pinned = torch.randn(6, 3).pin_memory()              # an already pinned source: the copy must not read it after the call returns
        want = pinned.clone()
        td = LS.to_dev(pinned, dev)
        pinned.zero_()
        check(torch.equal(td.cpu(), want), "to_dev: an already pinned tensor is copied before the call returns (the caller may change it)")
        iw = {}

        def make_index():
            with torch.inference_mode():
                iw["i"] = LS.index([3, 1, 2], dev)           # first made under inference_mode (the cache must not hold an inference tensor)
        make_index()
        xg = torch.randn(2, 5, requires_grad=True, device=dev)
        (xg[:, LS.index([3, 1, 2], dev)] * 2.0).sum().backward()
        check(xg.grad is not None and LS.index([3, 1, 2], dev) is iw["i"], "index: a tensor made under inference_mode still works in autograd")
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
    # the loss side on the GPU: the targets built with the switch on equal those with it off, on the GPU, with a per-sample weight (w_reg) and the object-consistency term
    gm = CachedBEVFusionModel(dataclasses.replace(cfg, sync_free=False)).eval().to(dev)
    gts_ = with_gt(batch_for(gm.cfg, B=Bn), gm.grid)["gt"]
    for i, g_ in enumerate(gts_):
        gts_[i] = {k: (dict(v, meta=dict(v.get("meta", {}), w_reg=0.37 + i)) if isinstance(v, dict) else v) for k, v in g_.items()}
    gcfg = L.LossConfig(use_w_reg=True)
    plain_t = L.build_layer_targets([g["cur"] for g in gts_], gcfg, None, 0.0, dev)
    with LS.enabled(sync=True):
        fast_t = L.build_layer_targets([g["cur"] for g in gts_], gcfg, None, 0.0, dev)
    same_t = set(plain_t) == set(fast_t) and all((torch.equal(plain_t[k], fast_t[k]) and fast_t[k].device.type == "cuda") if torch.is_tensor(plain_t[k]) else plain_t[k] == fast_t[k] for k in plain_t)
    one = L.build_layer_targets([dict(gts_[0]["cur"], meta=dict(gts_[0]["cur"]["meta"], w_reg=1.0))], gcfg, None, 0.0, dev)
    check(same_t and not torch.equal(one["w_lines"][0], plain_t["w_lines"][0]), f"build_layer_targets on the GPU: all {len(plain_t)} targets equal with the switch on and off, all on the GPU (w_reg of the samples 0.37 and 1.37 is in them)")
    ok_w = True
    for tgt in (plain_t, fast_t):                            # the weight of each sample is its own w_reg (an independent expectation, not on / off of the same code)
        for i in range(Bn):
            cells = tgt["w_map"][i] > 0
            ok_w &= bool(cells.any()) and torch.allclose(tgt["w_lines"][i][cells] / tgt["w_map"][i][cells], torch.full_like(tgt["w_map"][i][cells], 0.37 + i), rtol=1e-5)
    check(ok_w, "build_layer_targets on the GPU: w_lines = w_map x the sample's own w_reg (0.37, 1.37), switch on and off")
    tm = {"mask": torch.rand(2, 5), "ind": torch.randint(0, 20, (2, 5))}
    ov, vh = torch.randn(2, 2, 4, 5, device=dev), torch.randn(2, 2, 4, 5, device=dev)
    with LS.enabled(sync=True):
        c_fast = L.consistency_term(ov, vh, tm)
    mm, ii = tm["mask"].to(dev).unsqueeze(-1), tm["ind"].to(dev)
    c_ref = ((OH._gather(ov, ii) - OH._gather(vh, ii)).abs() * mm).sum() / (2 * mm.sum()).clamp(min=1)       # the formula written out, on the GPU
    check(torch.allclose(c_fast, c_ref, rtol=1e-6) and torch.equal(c_fast, L.consistency_term(ov, vh, tm)) and c_fast.device.type == "cuda",
          "consistency_term on the GPU: equals the formula written out, with the switch on and off")
    with LS.enabled(sync=True):
        tt_ = L._t(np.array([[1.0, 2.0]]), dev)
        t0_ = L._t(0.25, dev)
    check(tt_.device.type == "cuda" and t0_.device.type == "cuda" and tt_.tolist() == [[1.0, 2.0]] and t0_.item() == 0.25, "losses._t with the switch on: the tensor is on the GPU with its values (an array and a number)")
    # one real step on the GPU: the host copies are taken before the move and reach the model, and the sync-free loss and model still equal the plain ones
    gm.cfg.sync_free = True
    gbatch = distinct_calibrations(with_gt(batch_for(gm.cfg, B=Bn), gm.grid))
    gbatch["gap_steps"] = gap_steps(Bn)
    gbatch["flip"] = torch.tensor([True, False])
    gl = L.BEVLoss(grid=gm.grid)
    gl.cfg.sync_free = True
    gseen = {}
    _gf = gm.forward
    gm.forward = lambda b, *a, **k: (gseen.update(host=b.get("host"), dev=b["K"].device.type), _gf(b, *a, **k))[1]
    out_g, res_g, _ = T.forward_loss(gm, gl, gbatch, 0.5, dev, "off", None)
    gm.forward = _gf
    check(gseen["dev"] == "cuda" and gseen["host"] is not None and all(v.device.type == "cpu" for v in gseen["host"].values()) and gseen["host"]["K"] is gbatch["K"],
          "forward_loss on the GPU: the batch is on the GPU and batch['host'] holds the CPU tensors it was made from")
    gm.cfg.sync_free = False
    gl.cfg.sync_free = False
    out_p, res_p, _ = T.forward_loss(gm, gl, gbatch, 0.5, dev, "off", None)
    torch.cuda.synchronize()
    check(all(abs(float(out_g[k]) - float(out_p[k])) <= 1e-5 * max(1.0, abs(float(out_p[k]))) for k in out_p if k in out_g and torch.is_tensor(out_p[k]) and out_p[k].numel() == 1),
          "forward_loss on the GPU: every scalar loss term with host copies and sync_free equals the plain run to 1e-5 relative (the GPU scatter sums add in an order of its own)")
else:
    print("  skip  part 7 (no GPU)")

print("HOST BATCH GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)}): " + "; ".join(fails[:5])))
sys.exit(1 if fails else 0)
