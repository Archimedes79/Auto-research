"""Gate of the radar-only decoder (training/legacy/radar_decoder.py): stationary map + moving objects from radar alone.

Synthetic world (tests/synthetic_world.py): per sample a random scene at the output time; its past is reconstructed
with Scene.step(-age) for 6 sweeps at 12 Hz (ages 0 .. 5/12 s) and measured by 3 radars (front; front-left /
front-right rotated by +-80 deg as in nuScenes, roughly; 60 deg half field of view, 80 m; noise: range 0.1 m, azimuth
0.5 deg, Doppler 0.1 m/s - assumed values). Each detection is moved once to the output time (ego motion of its steps +
its radial Doppler motion, radar_decoder.to_output_time).
GT at the output time: 'static' = road edges (where the synthetic radar's static clutter lies; one cell wide) and the
footprints of objects slower than 0.5 m/s; 'moving objects' = boxes / circles of objects at >= 0.5 m/s. Everything is
evaluated inside the union of the radars' fields of view; moving objects are counted twice: those with at least one
radar return in the 6 sweeps (detectable) and all moving objects in the field of view.
Metrics on held-out scenes: static IoU at 0.5 (all, and per range band), moving objects AP at 0.5 / 1 / 2 m and the
match errors (tests/test_object_head.evaluate), recall at 2 m of all moving objects.
Arms: --sweeps 6 (default) or 1 (control); --no-doppler (detections moved by the ego motion only).
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_radar_only.py [--quick]
"""
import math
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "training")
sys.path.insert(0, "training/legacy")                 # radar_decoder (off the chain since 2026-10-31)
sys.path.insert(0, "tests")
from lift_pd import Grid  # noqa: E402
from object_head import GROUPS, build_targets, decode, head_loss  # noqa: E402
from radar_decoder import RadarOnlyDecoder, to_output_time  # noqa: E402
from synthetic_world import radar, random_scene  # noqa: E402
from test_object_head import evaluate, show  # noqa: E402

G = Grid()
DEV = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 3e9 else "cpu"
QUICK = "--quick" in sys.argv
SWEEPS = 1 if "--sweeps" in sys.argv and sys.argv[sys.argv.index("--sweeps") + 1] == "1" else 6
DOPPLER = "--no-doppler" not in sys.argv
RADARS = [(3.6, 0.0, 0.0), (3.4, 0.6, math.radians(80)), (3.4, -0.6, math.radians(-80))]
FOV, RMAX = math.radians(60), 80.0
NOISE = dict(sigma_r=0.1, sigma_phi_deg=0.5, sigma_d=0.1)
MOVING = 0.5

xs, ys = G.xy_centres()
GX, GY = torch.meshgrid(xs, ys, indexing="ij")


def fov_mask():
    m = torch.zeros(G.nx, G.ny, dtype=torch.bool)
    for sx, sy, yaw in RADARS:
        dx, dy = GX - sx, GY - sy
        az = torch.atan2(dy, dx) - yaw
        az = torch.atan2(torch.sin(az), torch.cos(az))
        m |= (az.abs() <= FOV) & (torch.hypot(dx, dy) <= RMAX)
    return m


MASK = fov_mask()


def box_cover(b, ss=4):
    o = (torch.arange(ss) + 0.5) / ss
    fx = G.x_min + (torch.arange(G.nx).view(-1, 1) + o).reshape(-1) * G.res
    fy = G.y_min + (torch.arange(G.ny).view(-1, 1) + o).reshape(-1) * G.res
    gx, gy = torch.meshgrid(fx, fy, indexing="ij")
    u = (gx - b.cx) * math.cos(b.heading) + (gy - b.cy) * math.sin(b.heading)
    w = -(gx - b.cx) * math.sin(b.heading) + (gy - b.cy) * math.cos(b.heading)
    return ((u.abs() <= b.length / 2) & (w.abs() <= b.width / 2)).float().view(G.nx, ss, G.ny, ss).mean((1, 3))


def sample(rng):
    scene = random_scene(rng)
    dets, seen = [], set()
    for j in range(SWEEPS):
        age = j / 12
        past = scene.step(-age) if age > 0 else scene
        steps = [(1 / 12, scene.v, scene.omega)] * j
        for pose in RADARS:
            d = radar(past, pose=pose, fov_deg=60.0, max_range=RMAX, rng=rng, **NOISE)
            if len(d["range"]) == 0:
                continue
            psi = d["azimuth"] + pose[2]
            pos, psi2 = to_output_time(d["xy"], psi, d["doppler"], steps, DOPPLER)
            dets.append({"x": pos[:, 0], "y": pos[:, 1], "r": d["range"], "psi": psi2, "vr": d["doppler"],
                         "rcs": d["rcs"], "dt": torch.full_like(d["range"], age)})
            seen |= {int(k) for k in d["box"].tolist() if k >= 0}
    det = {k: torch.cat([d[k] for d in dets]) for k in dets[0]}
    # GT: road edges (lateral coordinate in the lane frame, the radar clutter's lines) + slow objects; moving objects
    x0, y0, yaw = scene.lane_pose
    a = math.cos(yaw) * (GX - x0) + math.sin(yaw) * (GY - y0)
    yl = -math.sin(yaw) * (GX - x0) + math.cos(yaw) * (GY - y0)
    edges = [min(scene.lanes) - 1.5, max(scene.lanes) + 1.5]
    static = torch.zeros(G.nx, G.ny)
    for ye in edges:
        static = torch.maximum(static, ((yl - ye).abs() <= G.res / 2) & (a >= 5) & (a <= 60)).float()
    moving = []
    for k, b in enumerate(scene.boxes):
        if math.hypot(b.vx, b.vy) < MOVING:
            static = torch.maximum(static, box_cover(b))
        else:
            bx = {"group": b.cls, "cx": b.cx, "cy": b.cy, "vx": b.vx, "vy": b.vy, "seen": k in seen}
            if b.cls == "pedestrian":
                bx["radius"] = max(b.length, b.width) / 2
            else:
                bx.update(length=b.length, width=b.width, yaw=b.heading)
            moving.append(bx)
    return det, (static > 0.5).float(), moving


def in_fov(b):
    ix, iy = int((b["cx"] - G.x_min) / G.res), int((b["cy"] - G.y_min) / G.res)
    return 0 <= ix < G.nx and 0 <= iy < G.ny and bool(MASK[ix, iy])


def batch(items):
    dets = []
    for i, (d, _, _) in enumerate(items):
        dets.append({**d, "b": torch.full_like(d["x"], i, dtype=torch.long)})
    det = {k: torch.cat([d[k] for d in dets]).to(DEV) for k in dets[0]}
    static = torch.stack([s for _, s, _ in items]).to(DEV)
    return det, static, [[m for m in mv if m["seen"]] for _, _, mv in items]


def static_loss(logit, target):
    m = MASK.to(logit.device).float()
    bce = F.binary_cross_entropy_with_logits(logit[:, 0], target, pos_weight=torch.tensor(5.0, device=logit.device), reduction="none")
    return (bce * m).sum() / m.sum() / len(target)


def train(model, data, iters, bs, lr=2e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=iters, pct_start=0.1)
    model.train()
    t0 = time.time()
    for it in range(iters):
        det, static, moving = batch([data[random.randrange(len(data))] for _ in range(bs)])
        out = model(det, bs)
        lo = head_loss(out, build_targets(moving, G))
        ls = static_loss(out["static"], static)
        loss = lo["total"] + ls
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        sched.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            print(f"  iter {it:5d} objects {lo['total'].item():.3f} static {ls.item():.3f} ({time.time() - t0:.0f} s)", flush=True)


@torch.no_grad()
def run_eval(model, data, title):
    model.eval()
    dets, inter, union, band = [], 0.0, 0.0, {}
    for i in range(0, len(data), 8):
        items = data[i:i + 8]
        det, static, _ = batch(items)
        out = model(det, len(items))
        pred = (torch.sigmoid(out["static"][:, 0]) > 0.5).float().cpu()
        tgt = static.cpu()
        m = MASK.float()
        inter += (pred * tgt * m).sum().item()
        union += (((pred + tgt) > 0).float() * m).sum().item()
        for lo, hi in ((0, 20), (20, 40), (40, 60)):
            mb = m * ((GX >= lo) & (GX < hi)).float()
            ii, uu = band.get((lo, hi), (0.0, 0.0))
            band[(lo, hi)] = (ii + (pred * tgt * mb).sum().item(), uu + (((pred + tgt) > 0).float() * mb).sum().item())
        dets.extend(decode({k: v.cpu() for k, v in out.items() if k != "static"}, G, threshold=0.2))
    print(f"{title}\n  static map IoU (fields of view) {inter / max(1, union):.3f}; per range band " +
          ", ".join(f"{lo}-{hi} m {ii / max(1, uu):.3f}" for (lo, hi), (ii, uu) in band.items()))
    seen_gt = [[m for m in mv if m["seen"] and in_fov(m)] for _, _, mv in data]
    show(evaluate([[d for d in ds if in_fov(d)] for ds in dets], seen_gt), "  moving objects with >= 1 radar return")
    all_gt = [[m for m in mv if in_fov(m)] for _, _, mv in data]
    n_all = sum(len(g) for g in all_gt)
    n_seen = sum(len(g) for g in seen_gt)
    hit = 0
    for ds, gs in zip(dets, all_gt):
        used = set()
        for g in gs:
            for j, d in enumerate(ds):
                if j not in used and d["group"] == g["group"] and math.hypot(d["cx"] - g["cx"], d["cy"] - g["cy"]) <= 2.0:
                    used.add(j)
                    hit += 1
                    break
    print(f"  all moving objects in the fields of view: {n_all}, with a radar return {n_seen}; found within 2 m "
          f"(any score >= 0.2): {hit} = recall {hit / max(1, n_all):.3f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    print(f"device {DEV}, sweeps {SWEEPS}, Doppler motion {DOPPLER}")
    t0 = time.time()
    rng = random.Random(7)
    n_train, n_test = (200, 20) if QUICK else (1500, 60)
    train_set = [sample(rng) for _ in range(n_train)]
    rng = random.Random(4242)
    test_set = [sample(rng) for _ in range(n_test)]
    print(f"data: {n_train} training / {n_test} held-out scenes, {time.time() - t0:.0f} s; "
          f"mean detections per sample {sum(len(d['x']) for d, _, _ in train_set) / n_train:.0f}")
    one = train_set[:1]
    model = RadarOnlyDecoder(G).to(DEV)
    print("(1) overfit one scene")
    train(model, one, 150 if QUICK else 400, 1)
    run_eval(model, one, "    that scene")
    model = RadarOnlyDecoder(G).to(DEV)
    print(f"(2) train on {n_train} scenes")
    train(model, train_set, 400 if QUICK else 4000, 8)
    run_eval(model, test_set, "    held-out scenes")
    run_eval(model, train_set[:n_test], "    training scenes (same count)")
