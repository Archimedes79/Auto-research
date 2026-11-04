"""Gate of requirement R-GEO-1..4 (requirements.md): the camera projection is built from calibration data, is a GENERAL pinhole projection
(any rotation incl. roll, any position, fx != fy, off-centre principal point, skew, an augmentation transform with rotation / scale / crop), is
not learned, and needs no idealised virtual camera.

A  against an independent implementation (image position AND depth coordinate): for random calibrations (roll up to 35 deg, pitch 12, yaw 20, positions anywhere, fx != fy, principal
   point off centre, skew, an image transform with a 10 deg rotation, scale 0.5 and a crop) the image position of every lift voxel of
   `lift_pd.build_table` equals the position computed here from the 3 x 4 matrix P = K [R | t] (numpy, float64): max error below 0.02 px, depth below 1 cm.
B  oracle on rendered images: the synthetic camera (ray casting, exact depth) rolled by 0 / 20 / -25 / 35 deg, at a level pose and at a pitched, yawed,
   moved pose; the oracle lift (class features + exact depth histogram, the recommended lift) must rank the cells the rays hit at the top: R-precision
   >= 0.75 and no more than 0.15 below the same pose without roll. (The 5 %-of-max thresholded lane recall falls from 1.0 to 0.6 with roll - the
   mass spreads differently over axis-aligned 8 x 8 pixel groups - while the ranking stays at 0.97: a threshold effect, not a geometric error.)
D  the lift tables have no parameters (R-GEO-3).
C  the image transform enters as a plain 2D map: a camera image rotated by 90 deg together with its calibration (principal point and axes) gives
   the same BEV as the unrotated one (pure geometry: no learned part, no virtual camera).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_projection_general.py [--quick]
"""
import math
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
sys.path.insert(0, "tests")
from lift_pd import Grid, LiftPD, build_table  # noqa: E402
from synthetic_world import camera, random_scene  # noqa: E402
from test_oracle_lift import DC, depth_dist, features, hit_cells, lifted, range_recall, score  # noqa: E402

torch.set_num_threads(2)


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


R0 = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])                     # vehicle (x fwd, y left, z up) -> camera (x right, y down, z fwd)


def make_camera(roll=0.0, pitch=0.0, yaw=0.0, pos=(1.5, 0.0, 1.5), f=(560.0, 560.0), c=None, skew=0.0, hw=(256, 704)):
    """K (3 x 3), T_cam_car (4 x 4) for a camera at `pos` (vehicle frame) with the given roll about the optical axis, pitch (nose down > 0)
    and yaw (left > 0); float64 numpy."""
    H, W = hw
    c = c or ((W - 1) / 2, (H - 1) / 2)
    K = np.array([[f[0], skew, c[0]], [0, f[1], c[1]], [0, 0, 1.0]])
    # camera attitude: yaw about the vehicle z, pitch about the camera x (after R0), roll about the camera z
    R = rot("z", roll) @ rot("x", pitch) @ R0 @ rot("z", -yaw)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ np.array(pos)
    return K, T


def independent_pixels(pts, K, T, ida):
    """(N, 3) vehicle points -> (u, v, z_cam) with P = K [R | t] and the image transform, float64 numpy."""
    P = K @ T[:3, :]
    x = np.c_[pts, np.ones(len(pts))] @ P.T
    z = x[:, 2]
    uv = np.c_[x[:, 0] / z, x[:, 1] / z, np.ones(len(z))] @ ida.T
    return uv[:, 0], uv[:, 1], z


def test_A(n_cal=8, seed=0):
    rng = random.Random(seed)
    grid = Grid()
    worst, worst_z = 0.0, 0.0
    for _ in range(n_cal):
        hw = (256, 704)
        K, T = make_camera(roll=rng.uniform(-35, 35), pitch=rng.uniform(-12, 12), yaw=rng.uniform(-20, 20),
                           pos=(rng.uniform(-1, 3), rng.uniform(-1, 1), rng.uniform(1.0, 2.2)),
                           f=(rng.uniform(450, 700), rng.uniform(450, 700)), c=(rng.uniform(300, 400), rng.uniform(110, 150)),
                           skew=rng.uniform(-2, 2), hw=hw)
        a = math.radians(10)
        ida = np.array([[0.5 * math.cos(a), -0.5 * math.sin(a), rng.uniform(-30, 30)], [0.5 * math.sin(a), 0.5 * math.cos(a), rng.uniform(-30, 30)], [0, 0, 1.0]])
        z_levels = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=torch.float64)
        tab = build_table(grid, z_levels, torch.tensor(K), torch.tensor(T), (int(hw[0] * 0.8), int(hw[1] * 0.8)), DC.double(), torch.tensor(ida))
        xs, ys = grid.xy_centres(dtype=torch.float64)
        gx, gy, gz = torch.meshgrid(xs, ys, z_levels, indexing="ij")
        pts = torch.stack([gx, gy, gz], -1).reshape(-1, 3).numpy()
        u, v, z = independent_pixels(pts, K, T, ida)
        H2, W2 = int(hw[0] * 0.8), int(hw[1] * 0.8)
        un = (tab["grid2d"][:, 0].numpy() + 1) / 2 * W2 - 0.5                    # back to pixels of the transformed image
        vn = (tab["grid2d"][:, 1].numpy() + 1) / 2 * H2 - 0.5
        ok = tab["valid"].numpy() > 0
        assert ok.sum() > 1000, "enough visible voxels for a meaningful comparison"
        err = np.maximum(np.abs(un - u)[ok], np.abs(vn - v)[ok]).max()
        worst = max(worst, float(err))
        frac = ((tab["grid3d"][:, 2].numpy() + 1) * len(DC) - 1) / 2                # fractional depth-bin index of the table
        z_tab = DC[0].item() + frac * (DC[1] - DC[0]).item()
        worst_z = max(worst_z, float(np.abs(z_tab - z)[ok].max()))
    return worst, worst_z


def rprec(lift, hit):
    """threshold-free ranking quality: of the K = |hit cells| cells with the most lifted mass, the share within one cell of a hit cell
    (the thresholded recall depends on how the mass spreads, which changes with the roll without any geometric error)."""
    import torch.nn.functional as F
    k = int(hit.sum())
    if k == 0:
        return float("nan")
    top = torch.topk(lift.flatten(), k).indices
    near = F.max_pool2d(hit[None, None].float(), 3, 1, 1)[0, 0].flatten() > 0
    return float(near[top].float().mean())


def rolled_oracle(roll=0.0, pitch=0.0, yaw=0.0, pos=(1.5, 0.0, 1.5), n_scenes=4, seed=7):
    grid = Grid()
    rng = random.Random(seed)
    K, T = make_camera(roll, pitch, yaw, pos)
    Kt, Tt = torch.tensor(K), torch.tensor(T)
    hw = (256, 704)
    rec = {"lanes": [], "vehicles": [], "lanes_t": [], "vehicles_t": []}
    for _ in range(n_scenes):
        sc = random_scene(rng)
        img, depth, cls = camera(sc, Kt, Tt, hw)
        hits = hit_cells(depth, cls, sc, Kt, Tt, grid)
        lf = lifted(features(sc, cls, 8), depth_dist(depth, 8, "hist"), Kt, Tt, hw, grid, grid.z_levels, 4)
        s = score(lf, hits)
        rec["lanes"].append(rprec(lf[1], hits[1]))
        rec["vehicles"].append(rprec(lf[2], hits[2]))
        rec["lanes_t"].append(s[1][0])
        rec["vehicles_t"].append(s[2][0])
    return {k: float(np.nanmean(v)) for k, v in rec.items()}


def test_C():
    """the image rotated by 90 deg with its calibration gives the same lift: x' = H - 1 - v, y' = u (image of size H x W -> W x H)."""
    grid = Grid()
    hw = (256, 704)
    K, T = make_camera(roll=7.0, pitch=3.0, yaw=2.0, hw=hw)
    H, W = hw
    ida = np.array([[0, -1.0, H - 1], [1.0, 0, 0], [0, 0, 1]])                    # (u, v) -> (H - 1 - v, u): rotation by 90 deg
    z_levels = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    t1 = build_table(grid, z_levels, torch.tensor(K), torch.tensor(T), hw, DC.double())
    t2 = build_table(grid, z_levels, torch.tensor(K), torch.tensor(T), (W, H), DC.double(), torch.tensor(ida))
    ok = (t1["valid"] > 0) & (t2["valid"] > 0)
    assert ok.sum() > 1000
    # a pixel (u, v) of the original image sits at (H - 1 - v, u) in the rotated one: compare in the rotated image's pixels
    u1 = (t1["grid2d"][:, 0] + 1) / 2 * W - 0.5
    v1 = (t1["grid2d"][:, 1] + 1) / 2 * H - 0.5
    u2 = (t2["grid2d"][:, 0] + 1) / 2 * H - 0.5                                   # rotated image: width H, height W
    v2 = (t2["grid2d"][:, 1] + 1) / 2 * W - 0.5
    return float(torch.maximum((u2 - (H - 1 - v1)).abs()[ok], (v2 - u1).abs()[ok]).max()), int(ok.sum())


def test_D():
    """R-GEO-3: the projection tables have nothing to learn (no parameters) unless learned heights are switched on, and then only the heights."""
    from fallbacks import LiftS
    grid = Grid()
    assert sum(p.numel() for p in LiftPD(grid, DC).parameters()) == 0
    assert sum(p.numel() for p in LiftPD(grid, DC, use_depth=False).parameters()) == 0
    assert sum(p.numel() for p in LiftS(grid, DC).parameters()) == 0
    learned = LiftPD(grid, DC, learn_heights=True)
    assert [n for n, _ in learned.named_parameters()] == ["z_levels"]
    return True


def main():
    quick = "--quick" in sys.argv
    errA, errZ = test_A(3 if quick else 8)
    print(f"A  independent projection (roll up to 35 deg, pitch 12, yaw 20, fx != fy, off-centre principal point, skew, image transform with 10 deg "
          f"rotation + scale + crop): worst image-position error of any voxel {errA:.4f} px, worst depth error {errZ:.4f} m")
    errC, nC = test_C()
    print(f"C  image rotated by 90 deg with its calibration: worst difference {errC:.4f} px over {nC} voxels")
    ok_d = test_D()
    print("D  the lift tables have no parameters (lift PD, P, S); learned heights, if switched on, are the only parameter")
    poses = {"level camera pose": {}, "pitch 6, yaw 8, other position": dict(pitch=6.0, yaw=8.0, pos=(0.3, 0.4, 1.9))}
    rolls = (0.0, 20.0) if quick else (0.0, 20.0, -25.0, 35.0)
    ok = errA < 0.02 and errZ < 0.01 and errC < 0.02 and ok_d
    print("B  oracle lift on rendered images (exact depth, stride 8, oversampling 4, depth histogram), ranking quality = R-precision "
          "(thresholded recall in brackets):")
    for pn, pose in poses.items():
        base = None
        for roll in rolls:
            r = rolled_oracle(roll=roll, **pose, n_scenes=3 if quick else 8)
            base = base or r
            print(f"     {pn:32s} roll {roll:5.0f}: lanes {r['lanes']:.2f} ({r['lanes_t']:.2f}), vehicles {r['vehicles']:.2f} ({r['vehicles_t']:.2f})")
            ok &= all(r[k] >= 0.75 and r[k] >= base[k] - 0.15 for k in ("lanes", "vehicles"))
    print(f"GATE projection (requirements R-GEO): {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
