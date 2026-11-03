"""Synthetic data of the camera-only S1 pipeline in the format of the REAL feature cache (plan.md 10.1 / experiments/s1a_lift_pd_vs_p, section 3).

One sample = one random synthetic scene (tests/support/synthetic_world.py) seen by the front camera with the geometry of the
real post-ida image (256 x 704, f = 560, 1.5 m high, feature map 16 x 44 = stride 16, the resolution of DualBEV's
features - the oracle gate at stride 8 was too kind, plan 10.1):
- ctx (80, 16, 44): what the image encoder would know without any depth: the pixel classes of every 16 x 16 group as
  fractions (sky, road, lane line, vehicle, pedestrian) + 75 noise channels; perfect semantics, unknown position
- prob_oracle (118, 16, 44): the depth head's output if it were perfect: the histogram of the group's pixel depths
  over the 118 bins (tests/test_oracle_lift.depth_hist); variants made at batch time by `depth_variant`:
  blurred (Gaussian in 1/z, mimics the frozen head's far-range blur), shuffled (another sample's: keeps the scene-independent flat-ground depth, see the S1a card section 5), permuted (the sample's histograms at random other pixel groups), uniform
- gt (3, 120, 104): lane lines, vehicles, pedestrians as coverage on the BEV grid (synthetic_world.gt_bev); mask (120, 104):
  the cells the camera sees (the cells whose height samples project into the image, LiftPD.visibility), float
- K, T_cam_car, hw: the rig, the same for every sample; ida = None
Lane lines vary per sample (offset, spacing, yaw): a fixed position would let a per-cell prior solve them.
Stored in `temp/cache/s1_synth.pt` (derived data, not in git). Build: python studies/early/synthetic_s1_data.py build [n_train n_val] (without arguments: the self-test)
The real data gets the same dict layout (prob, ctx, K, T_cam_car, hw, gt, mask, scene) in the S1a driver (to build, from
training/feature_cache.py and training/gt_bev.py).
"""
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))     # the bev_fusion folder
sys.path.insert(0, os.path.join(ROOT, "training"))
sys.path.insert(0, os.path.join(ROOT, "tests"))                           # test_oracle_lift (a gate)
sys.path.insert(0, os.path.join(ROOT, "tests", "support"))
from lift_pd import Grid, LiftPD  # noqa: E402
from synthetic_world import camera, front_camera, gt_bev, random_scene  # noqa: E402
from test_oracle_lift import DC, depth_hist  # noqa: E402

CACHE = os.path.join(ROOT, "temp", "cache", "s1_synth.pt")
CLASS_NAMES = ("lanes", "vehicles", "pedestrians")
N_CTX, STRIDE = 80, 16
# coverage fraction of a 0.5 m cell above which a class is present: a lane line (15 cm wide) touches a cell with a fraction
# of about 0.3 - the real GT's line mask is a 1-cell mask, not a coverage; a pedestrian (0.6 m) fills a cell at most 0.36
GT_THRESHOLDS = (0.1, 0.5, 0.25)
SIGMA_INV = 0.01                      # blurred depth: Gaussian width in 1/z (1/m): 1 m at 10 m, 9 m at 30 m, 36 m at 60 m


def make_sample(seed):
    rng = random.Random(seed)
    scene = random_scene(rng)
    y0, w = rng.uniform(-1.5, 1.5), rng.uniform(3.2, 4.0)
    scene.lanes = [y0 + (k - 1.5) * w for k in range(4)]
    scene.lane_pose = (0.0, 0.0, rng.uniform(-0.08, 0.08))
    K, T, hw = front_camera()
    img, depth, cls = camera(scene, K, T, hw)
    grid = Grid()
    gt = gt_bev(scene, grid)
    ped = torch.zeros_like(cls, dtype=torch.bool)
    veh = torch.zeros_like(ped)
    for k, b in enumerate(scene.boxes):
        (ped if b.cls == "pedestrian" else veh).__ior__(cls == 3 + k)
    sky = ~torch.isfinite(depth)
    onehot = torch.stack([sky, (cls == 1) & ~sky, cls == 2, veh, ped]).float()
    pooled = F.avg_pool2d(onehot[None], STRIDE)[0]                                  # (5, 16, 44) fractions
    g = torch.Generator().manual_seed(seed)
    ctx = torch.cat([pooled, 0.3 * torch.randn(N_CTX - 5, *pooled.shape[1:], generator=g)])
    prob = depth_hist(depth, STRIDE)                                                # (118, 16, 44)
    mask = LiftPD(grid, DC, oversample=4).visibility(K.float(), T.float(), hw)
    return {"ctx": ctx.half(), "prob": prob.half(), "gt": torch.stack([gt[n] for n in CLASS_NAMES]).half(),
            "mask": mask.half(), "seed": seed}


def build(n_train=300, n_val=60, path=CACHE, workers=6):
    import multiprocessing as mp
    seeds = [1000 + i for i in range(n_train)] + [900000 + i for i in range(n_val)]
    with mp.Pool(workers, initializer=lambda: torch.set_num_threads(1)) as pool:
        samples = pool.map(make_sample, seeds, chunksize=4)
    K, T, hw = front_camera()
    data = {"train": samples[:n_train], "val": samples[n_train:], "K": K.float(), "T_cam_car": T.float(), "hw": hw}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    return data


def load(path=CACHE):
    return torch.load(path, weights_only=True) if os.path.exists(path) else build(path=path)


def blur_matrix(sigma_inv=SIGMA_INV):
    """(D, D) matrix M: column i = a Gaussian in 1/z around bin centre i, normalised over the bins (a depth head that is
    sharp near and blurred far, like the frozen DualBEV head: its depth error is about constant in 1/z, plan 2.4)."""
    inv = 1.0 / DC
    M = torch.exp(-0.5 * ((inv.view(-1, 1) - inv.view(1, -1)) / sigma_inv) ** 2)
    return M / M.sum(0, keepdim=True)


_BLUR = None


def depth_variant(prob, kind):
    """prob (B, D, h, w) oracle histograms -> the variant: 'oracle', 'blurred', 'shuffled' (the next sample's), 'permuted', 'uniform'."""
    global _BLUR
    if kind == "oracle":
        return prob
    if kind == "blurred":
        _BLUR = _BLUR if _BLUR is not None else blur_matrix()
        out = torch.einsum("ji,bihw->bjhw", _BLUR.to(prob), prob)
        return out / out.sum(1, keepdim=True).clamp(min=1e-6) * (prob.sum(1, keepdim=True) > 0)
    if kind == "shuffled":
        return prob.roll(1, 0)
    if kind == "permuted":                    # the same histograms at random other pixel groups of the SAME sample: marginal kept, depth-pixel relation destroyed
        B, D_, h, w = prob.shape
        g = torch.Generator().manual_seed(1234)
        out = prob.clone().view(B, D_, h * w)
        for i in range(B):
            out[i] = out[i][:, torch.randperm(h * w, generator=g)]
        return out.view(B, D_, h, w)
    if kind == "uniform":
        return torch.full_like(prob, 1.0 / prob.shape[1])
    raise ValueError(kind)


def batch(samples, data, kind="oracle", device="cpu"):
    """list of samples -> the dict the S1 module reads (training/s1_camera.CameraBEVSeg.forward / train_s1)."""
    f = lambda k: torch.stack([s[k] for s in samples]).float().to(device)       # noqa: E731
    B = len(samples)
    thr = torch.tensor(GT_THRESHOLDS, device=device).view(1, -1, 1, 1)
    gt = (f("gt") > thr).float()                                                # presence masks, as the real GT's line mask
    return {"prob": depth_variant(f("prob"), kind), "ctx": f("ctx"), "gt": gt, "mask": f("mask")[:, None],
            "K": data["K"].to(device).expand(B, 3, 3), "T_cam_car": data["T_cam_car"].to(device).expand(B, 4, 4),
            "hw": data["hw"], "ida": None, "scene": [s["seed"] for s in samples]}


def _selftest():
    s = make_sample(7)
    assert s["ctx"].shape == (N_CTX, 16, 44) and s["prob"].shape == (118, 16, 44) and s["gt"].shape == (3, 120, 104)
    p = s["prob"].float()
    nonsky = p.sum(0) > 0
    assert torch.allclose(p.sum(0)[nonsky], torch.ones(int(nonsky.sum())), atol=2e-3), "depth histograms sum to 1"
    assert s["gt"][1].float().sum() > 0 and s["mask"].float().sum() > 500, "vehicles and a visible area exist"
    M = blur_matrix()
    assert torch.allclose(M.sum(0), torch.ones(118), atol=1e-5)
    near, far = M[:, 4], M[:, 100]                                                 # bins at 3.25 m and 51.25 m
    sd = lambda c: float(torch.sqrt((c * (DC - (c * DC).sum()) ** 2).sum()))      # noqa: E731
    assert sd(far) > 10 * sd(near), (sd(near), sd(far))
    b = depth_variant(p[None], "blurred")
    assert torch.allclose(b.sum(1)[0][nonsky], torch.ones(int(nonsky.sum())), atol=1e-4)
    print(f"synthetic_s1_data selftest ok: shapes of the real cache format, histograms sum to 1, blur sd {sd(near):.1f} m "
          f"at 3 m and {sd(far):.1f} m at 51 m, {int(s['gt'][1].float().sum())} vehicle cells, "
          f"{int(s['mask'].float().sum())} visible cells")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        n_tr, n_va = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (300, 60)
        d = build(n_tr, n_va)
        print(f"built {len(d['train'])} train / {len(d['val'])} val samples -> {CACHE}")
    else:
        _selftest()
