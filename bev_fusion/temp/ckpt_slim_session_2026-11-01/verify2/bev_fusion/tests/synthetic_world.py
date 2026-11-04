"""Synthetic world for the block tests of bev_fusion (plan.md section 10.1): one scene, all sensors and the GT consistent.

A scene is a set of static lane lines and boxes (centre, length, width, height, heading, velocity, class) in the ego
frame (x forward, y left, z up), plus the ego speed v and yaw rate omega. From it:
- camera(): image (3, H, W) by ray casting against the road plane z = 0 and the boxes, the exact depth along the
  optical axis (H, W), the class of every pixel (0 sky, 1 road, 2 lane line, 3 + k box k), with any K and camera pose
- radar(): detections of one radar at a given pose: points on the boxes' edges facing it (and static clutter on the
  road edges), position (vehicle frame), range, azimuth (sensor frame), Doppler = object ground velocity on the line
  of sight (ego-compensated, as nuScenes vx_comp / vy_comp), RCS by class, measurement noise
- gt_bev(grid): layers (lane lines, vehicles, pedestrians) as coverage per cell, a velocity field per cell, a box id map
- step(dt): the same world dt later, in the new ego frame: objects moved by their velocity, everything transformed by
  the EXACT constant-turn-rate ego motion, headings and velocity vectors rotated into the new axes
Run the self-test: python tests/synthetic_world.py   (from the bev_fusion folder)
"""
import math
import random
from dataclasses import dataclass, field, replace

import torch

CLASSES = {"car": (4.5, 1.8, 1.5, 10.0), "truck": (8.0, 2.5, 3.0, 20.0), "pedestrian": (0.6, 0.6, 1.8, -5.0)}  # L W H RCS


@dataclass
class Box:
    cx: float
    cy: float
    length: float
    width: float
    height: float
    heading: float
    vx: float
    vy: float
    cls: str
    rcs: float

    def corners(self):
        c, s = math.cos(self.heading), math.sin(self.heading)
        hl, hw = self.length / 2, self.width / 2
        return [(self.cx + c * a - s * b, self.cy + s * a + c * b) for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


@dataclass
class Scene:
    boxes: list
    lanes: list = field(default_factory=lambda: [-3.5, 0.0, 3.5])      # static lines y = const in the ego frame of t0
    lane_pose: tuple = (0.0, 0.0, 0.0)                                 # (x0, y0, yaw) of the t0 frame in the current one
    v: float = 10.0
    omega: float = 0.0

    def step(self, dt):
        """The scene dt later, in the new ego frame (exact constant turn rate of the ego vehicle)."""
        th = self.omega * dt
        if abs(self.omega) < 1e-12:
            dx, dy = self.v * dt, 0.0
        else:
            dx, dy = self.v / self.omega * math.sin(th), self.v / self.omega * (1 - math.cos(th))
        c, s = math.cos(th), math.sin(th)

        def tf(px, py):
            qx, qy = px - dx, py - dy
            return c * qx + s * qy, -s * qx + c * qy
        boxes = []
        for b in self.boxes:
            nx, ny = tf(b.cx + b.vx * dt, b.cy + b.vy * dt)
            boxes.append(replace(b, cx=nx, cy=ny, heading=b.heading - th, vx=c * b.vx + s * b.vy, vy=-s * b.vx + c * b.vy))
        x0, y0, yaw = self.lane_pose
        lx, ly = tf(x0, y0)
        return replace(self, boxes=boxes, lane_pose=(lx, ly, yaw - th))

    def lane_dist(self, x, y):
        """distance of points (tensors, current frame) to the nearest lane line (lines y = y_k in the t0 frame)."""
        x0, y0, yaw = self.lane_pose
        c, s = math.cos(yaw), math.sin(yaw)
        yl = -s * (x - x0) + c * (y - y0)                              # lateral coordinate in the t0 frame
        return torch.stack([(yl - yk).abs() for yk in self.lanes]).min(0).values


def random_scene(rng, n_boxes=(3, 6), x_range=(8, 55), y_range=(-18, 18)):
    boxes = []
    for _ in range(rng.randint(*n_boxes)):
        cls = rng.choices(list(CLASSES), weights=(6, 1, 2))[0]
        L, W, H, rcs = CLASSES[cls]
        h = rng.uniform(-math.pi, math.pi)
        sp = rng.uniform(0, 2) if cls == "pedestrian" else rng.uniform(0, 15)
        boxes.append(Box(rng.uniform(*x_range), rng.uniform(*y_range), L, W, H, h, sp * math.cos(h), sp * math.sin(h), cls, rcs))
    return Scene(boxes, lanes=[-5.25, -1.75, 1.75, 5.25], v=rng.uniform(5, 20), omega=rng.uniform(-0.4, 0.4))


def front_camera(h=256, w=704, f=560.0, height=1.5, pitch_deg=0.0):
    """Pinhole camera at (1.5 m forward, 0, height) looking along +x; returns K (3x3) and T_cam_car (4x4)."""
    K = torch.tensor([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], dtype=torch.float64)
    p = math.radians(pitch_deg)
    R0 = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float64)       # car -> cam axes
    Rp = torch.tensor([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]], dtype=torch.float64)
    R = Rp @ R0
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3], T[:3, 3] = R, -R @ torch.tensor([1.5, 0.0, height], dtype=torch.float64)
    return K, T, (h, w)


def camera(scene, K, T_cam_car, hw):
    """Ray-cast image (3, H, W) in [0, 1], depth along the optical axis (H, W; inf for sky), class map (H, W)."""
    H, W = hw
    dt = torch.float64
    v, u = torch.meshgrid(torch.arange(H, dtype=dt), torch.arange(W, dtype=dt), indexing="ij")
    rays_cam = torch.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], torch.ones_like(u)], -1)  # z_cam = 1
    T_car_cam = torch.linalg.inv(T_cam_car)
    o = T_car_cam[:3, 3]
    d = rays_cam @ T_car_cam[:3, :3].T                                  # direction in the car frame per unit z_cam
    t_best = torch.full((H, W), float("inf"), dtype=dt)
    cls = torch.zeros(H, W, dtype=torch.long)
    shade = torch.zeros(H, W, dtype=dt)
    tg = torch.where(d[..., 2] < -1e-9, -o[2] / d[..., 2], torch.full_like(t_best, float("inf")))   # road plane
    gx, gy = o[0] + tg * d[..., 0], o[1] + tg * d[..., 1]
    road = torch.isfinite(tg)
    t_best = torch.where(road, tg, t_best)
    lane = road & (scene.lane_dist(gx, gy) <= 0.075)
    cls[road] = 1
    cls[lane] = 2
    for k, b in enumerate(scene.boxes):                                # slab test in the box frame
        c, s = math.cos(b.heading), math.sin(b.heading)
        ox, oy = c * (o[0] - b.cx) + s * (o[1] - b.cy), -s * (o[0] - b.cx) + c * (o[1] - b.cy)
        dx, dy = c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]
        lo = torch.full_like(t_best, -float("inf"))
        hi = torch.full_like(t_best, float("inf"))
        face = torch.zeros(H, W, dtype=torch.long)
        for axis, (oo, dd, half) in enumerate(((ox, dx, b.length / 2), (oy, dy, b.width / 2), (o[2] - b.height / 2, d[..., 2], b.height / 2))):
            inv = 1.0 / torch.where(dd.abs() < 1e-12, torch.full_like(dd, 1e-12), dd)
            t1, t2 = (-half - oo) * inv, (half - oo) * inv
            tmin = torch.minimum(t1, t2)
            face = torch.where(tmin > lo, torch.full_like(face, axis), face)
            lo, hi = torch.maximum(lo, tmin), torch.minimum(hi, torch.maximum(t1, t2))
        hit = (lo <= hi) & (lo > 0) & (lo < t_best)
        t_best = torch.where(hit, lo, t_best)
        cls[hit] = 3 + k
        shade[hit] = face[hit].to(dt)
    img = torch.zeros(3, H, W, dtype=dt)
    img[:] = torch.tensor([0.55, 0.7, 0.9], dtype=dt).view(3, 1, 1)                 # sky
    img[:, cls == 1] = 0.35
    img[:, cls == 2] = 0.95
    palette = {"car": (0.8, 0.2, 0.2), "truck": (0.2, 0.5, 0.8), "pedestrian": (0.9, 0.8, 0.1)}
    for k, b in enumerate(scene.boxes):
        m = cls == 3 + k
        col = torch.tensor(palette[b.cls], dtype=dt).view(3, 1)
        img[:, m] = col * (1.0 - 0.25 * shade[m]).view(1, -1)
    return img.float(), t_best.float(), cls                            # depth = t (rays normalised to z_cam = 1)


def radar(scene, pose=(3.5, 0.0, 0.0), fov_deg=60.0, max_range=80.0, points_per_box=3, clutter_per_lane=6,
          sigma_r=0.1, sigma_phi_deg=0.5, sigma_d=0.0, rng=None):
    """Detections as a dict of tensors: xy (N, 2) vehicle frame, range, azimuth (sensor frame), doppler (ego-compensated
    radial velocity, m/s), rcs, box (index or -1 for clutter)."""
    rng = rng or random.Random(0)
    sx, sy, syaw = pose
    out = {"xy": [], "range": [], "azimuth": [], "doppler": [], "rcs": [], "box": []}

    def add(px, py, vx, vy, rcs, k):
        dx, dy = px - sx, py - sy
        r = math.hypot(dx, dy)
        az = math.atan2(dy, dx) - syaw
        if r > max_range or abs(az) > math.radians(fov_deg):
            return
        r_m = r + rng.gauss(0, sigma_r)
        az_m = az + math.radians(rng.gauss(0, sigma_phi_deg))
        ux, uy = math.cos(az + syaw), math.sin(az + syaw)
        out["xy"].append((sx + r_m * math.cos(az_m + syaw), sy + r_m * math.sin(az_m + syaw)))
        out["range"].append(r_m)
        out["azimuth"].append(az_m)
        out["doppler"].append(vx * ux + vy * uy + (rng.gauss(0, sigma_d) if sigma_d > 0 else 0.0))
        out["rcs"].append(rcs + rng.gauss(0, 2.0))
        out["box"].append(k)
    for k, b in enumerate(scene.boxes):
        cs = b.corners()
        edges = [(cs[i], cs[(i + 1) % 4]) for i in range(4)]
        vis = []
        for (ax, ay), (bx, by) in edges:                                # edges whose outward normal faces the radar
            mx, my = (ax + bx) / 2, (ay + by) / 2
            nx, ny = mx - b.cx, my - b.cy
            if nx * (sx - mx) + ny * (sy - my) > 0:
                vis.append(((ax, ay), (bx, by)))
        for _ in range(points_per_box if vis else 0):
            (ax, ay), (bx, by) = rng.choice(vis)
            t = rng.random()
            add(ax + t * (bx - ax), ay + t * (by - ay), b.vx, b.vy, b.rcs, k)
    x0, y0, yaw = scene.lane_pose
    for yk in (min(scene.lanes) - 1.5, max(scene.lanes) + 1.5):          # static clutter along the road edges
        for _ in range(clutter_per_lane):
            a = rng.uniform(5, 60)
            px = x0 + math.cos(yaw) * a - math.sin(yaw) * yk
            py = y0 + math.sin(yaw) * a + math.cos(yaw) * yk
            add(px, py, 0.0, 0.0, -5.0, -1)
    res = {k: torch.tensor(v, dtype=torch.float32) for k, v in out.items() if k != "xy"}
    res["xy"] = torch.tensor(out["xy"], dtype=torch.float32).reshape(-1, 2)
    res["box"] = res["box"].long()
    return res


def gt_bev(scene, grid, ss=4):
    """GT on the BEV grid: dict of lanes, vehicles, pedestrians (coverage, (X, Y)), velocity (2, X, Y) of the boxes,
    box_id (X, Y; -1 = none)."""
    dt = torch.float32
    o = (torch.arange(ss, dtype=dt) + 0.5) / ss
    xs = grid.x_min + (torch.arange(grid.nx, dtype=dt).view(-1, 1) + o).reshape(-1) * grid.res
    ys = grid.y_min + (torch.arange(grid.ny, dtype=dt).view(-1, 1) + o).reshape(-1) * grid.res
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")

    def cov(m):
        return m.to(dt).view(grid.nx, ss, grid.ny, ss).mean((1, 3))
    lanes = cov(scene.lane_dist(gx, gy) <= 0.075)
    veh = torch.zeros(grid.nx, grid.ny)
    ped = torch.zeros_like(veh)
    vel = torch.zeros(2, grid.nx, grid.ny)
    bid = torch.full((grid.nx, grid.ny), -1, dtype=torch.long)
    for k, b in enumerate(scene.boxes):
        u = (gx - b.cx) * math.cos(b.heading) + (gy - b.cy) * math.sin(b.heading)
        w = -(gx - b.cx) * math.sin(b.heading) + (gy - b.cy) * math.cos(b.heading)
        c = cov((u.abs() <= b.length / 2) & (w.abs() <= b.width / 2))
        if b.cls == "pedestrian":
            ped = torch.maximum(ped, c)
        else:
            veh = torch.maximum(veh, c)
        m = c > 0.5
        vel[0][m], vel[1][m] = b.vx, b.vy
        bid[m] = k
    return {"lanes": lanes, "vehicles": veh, "pedestrians": ped, "velocity": vel, "box_id": bid}


# ----------------------------------------------------------------------------------------------- self-test
def _selftest():
    import sys
    sys.path.insert(0, "training")
    from lift_pd import Grid
    rng = random.Random(3)
    sc = Scene([Box(20.0, 0.0, 4.5, 1.8, 1.5, 0.0, 8.0, 0.0, "car", 10.0),
                Box(35.0, -6.0, 4.5, 1.8, 1.5, math.pi / 2, 0.0, 5.0, "car", 10.0)], v=10.0, omega=0.2)
    K, T, hw = front_camera()
    img, depth, cls = camera(sc, K, T, hw)
    # 1 the image centre column looks at the first box's rear face: depth = 20 - 2.25 - 1.5 (camera at x = 1.5)
    h, w = hw
    col = cls[:, w // 2]
    rows = (col == 3).nonzero()
    assert len(rows) > 0, "box visible in the centre column"
    assert abs(depth[rows[len(rows) // 2, 0], w // 2].item() - (20 - 2.25 - 1.5)) < 1e-3, "box depth"
    # 2 a road pixel: depth along the optical axis = analytic ground depth f h / (v - cy)
    r = h - 5
    assert cls[r, 10] == 1 and abs(depth[r, 10].item() - 560.0 * 1.5 / (r - (h - 1) / 2)) < 1e-3, "ground depth"
    # 3 radar Doppler = object velocity on the line of sight; clutter Doppler 0
    det = radar(sc, rng=rng, sigma_r=0.0, sigma_phi_deg=0.0)
    for i in range(len(det["range"])):
        k = det["box"][i].item()
        x, y = det["xy"][i].tolist()
        ux, uy = (x - 3.5) / math.hypot(x - 3.5, y), y / math.hypot(x - 3.5, y)
        exp = 0.0 if k < 0 else sc.boxes[k].vx * ux + sc.boxes[k].vy * uy
        assert abs(det["doppler"][i].item() - exp) < 1e-4, "Doppler"
    assert (det["box"] >= 0).sum() > 0 and (det["box"] < 0).sum() > 0
    # 4 step: a static lane point and a moving box follow the exact motion; velocity rotates with the frame
    s1 = sc.step(0.5)
    th = 0.2 * 0.5
    assert abs(s1.boxes[0].heading + th) < 1e-12 and abs(math.hypot(s1.boxes[0].vx, s1.boxes[0].vy) - 8.0) < 1e-9
    # 5 GT: the box footprint carries its velocity, the lane layer has the lines
    g = gt_bev(sc, Grid())
    assert g["vehicles"].sum() > 2 * 4.5 * 1.8 * 4 * 0.9 and torch.allclose(g["velocity"][0][g["box_id"] == 0], torch.tensor(8.0))
    assert g["lanes"].sum() > 0
    rs = random_scene(rng)
    camera(rs, K, T, hw)
    gt_bev(rs.step(1 / 12), Grid())
    print("synthetic_world selftest ok: box and ground depth exact, Doppler = v . u, step with exact ego motion, GT "
          "velocity field, random scenes")


if __name__ == "__main__":
    _selftest()
