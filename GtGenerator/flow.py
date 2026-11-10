"""autoGT box: velocity / flow ground truth from tracked boxes, ego poses and (optionally) radar Doppler.

ONE function does the work:

    flow_gt = estimate_flow(tracks, ego, t, grid, radar=(), config=Config())

Everything is in SI units with REAL time stamps in seconds - velocities are m/s, never "per frame"; a displacement over any
dt is `flow * dt` (`displace`). No dataset knowledge lives here: a dataset adapter (e.g. `nuscenes_adapter.py`) only turns
its records into the four small inputs below. numpy only.

Inputs
  tracks  list[Track]      boxes over time: id, t (s, increasing), xy (world, m), yaw, size (w, l, h), group, observed
  ego     Ego              planar ego poses over time: t (s), x, y, yaw (world <- ego), any rate
  t       float            the frame time (s) the flow is wanted at; the grid is in the ego frame AT t (x forward, y left)
  grid    GridSpec | None  cell grid; None = object list only (no dense field)
  radar   list[RadarFrame] optional: radar detections already compensated for the ego motion: world position and the
                           radial speed (m/s, ground-relative, positive = moving away from the sensor)
What it does (the numbered steps are the whole algorithm)
  1. ego velocity (vx, vy) and yaw rate omega at t from the poses t +- ego_half_window (central difference over real time)
  2. per track: stationary? (rule of 2026-10-26: all central-difference speeds < v_stat and all positions < d_stat from the
     median) -> one fixed world pose, velocity 0.  Else velocity = slope of the least-squares line through the observations
     within +-window seconds of t (equal weights; the radar check chose it over the devkit's two-neighbour difference), with
     the slope's covariance sigma_pos^2 / sum (tau - mean tau)^2
  3. radar correction: the detections inside a box (box moved to the radar time, grown by gate_margin) measure the radial
     component r_hat . v; a robust Bayesian update (information form, 3 sigma gate, 2 passes, points weighted so that many
     detections of one box cannot claim more than `radar_cap` independent ones) refines the track velocity in the radial
     direction only and shrinks its covariance there
  4. flags per object: radar_conflict (most detections reject the track velocity: suspect annotation or association) and, when
     those detections agree among themselves (robust spread < 0.7 m/s), radar_override: the speed along the track direction
     is taken from the radar (Huber fit) - stop-and-go and reversed tracks,
     stationary_but_moving (a "stationary" box whose radar Doppler is clearly not zero), short_track (< 3 observations,
     no velocity known), unobserved (no sensor point in the box: weight 0)
  5. dense field in the ego frame at t:  flow_ego = -(v_ego + omega x r) for every static world point; flow_obj = the box's
     world velocity in ego axes on its footprint; flow = flow_ego + flow_obj (first order; see plan 7.1: the splat block
     uses the exact chain); weight 0 where the flow is unknown; sigma = standard deviation of the speed estimate (m/s)
Output FlowGT: dense arrays (nx, ny) / (2, nx, ny) in float32 plus `objects` (list[ObjectFlow]: pose at t, v_world, cov,
source, n_radar, flag) - the objects are the AutoGT product, the arrays are their rasterisation.
Self-test (synthetic world with known truth): python -m autogt.flow  (all tests of the suite: python -m autogt selftest)
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .grid import GridSpec
from .layers import cell_centres, inside_quad as _inside_quad


# --------------------------------------------------------------------------------------------------- inputs / outputs
@dataclass
class Track:
    id: str
    t: np.ndarray                     # (N,) s, increasing
    xy: np.ndarray                    # (N, 2) world, m
    yaw: np.ndarray                   # (N,) rad
    size: np.ndarray                  # (3,) width, length, height in m
    group: str = "vehicle"
    observed: np.ndarray = None       # (N,) bool: box has sensor points; None = all True
    stationary: bool = None           # None = decide from the track
    extra: dict = field(default_factory=dict)   # free adapter data (e.g. annotation tokens, 3D positions)

    def __post_init__(self):
        self.t, self.xy, self.yaw = np.asarray(self.t, float), np.asarray(self.xy, float), np.asarray(self.yaw, float)
        self.size = np.asarray(self.size, float)
        self.observed = np.ones(len(self.t), bool) if self.observed is None else np.asarray(self.observed, bool)


@dataclass
class Ego:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    def __post_init__(self):
        self.t, self.x, self.y = np.asarray(self.t, float), np.asarray(self.x, float), np.asarray(self.y, float)
        self.yaw = np.unwrap(np.asarray(self.yaw, float))

    def pose(self, t):
        return (float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)), float(np.interp(t, self.t, self.yaw)))


@dataclass
class RadarFrame:
    t: float
    sensor: str
    sensor_xy: np.ndarray             # (2,) world position of the sensor
    xy: np.ndarray                    # (K, 2) world positions of the detections
    doppler: np.ndarray               # (K,) radial speed m/s, ground-relative (ego motion removed), + = away from the sensor


@dataclass
class Config:
    window: float = 1.2               # s, observations used for the line fit
    sigma_pos: float = 0.2            # m, position noise of one annotation
    sigma_doppler: float = 0.5        # m/s, per detection (conservative: the measured median error is 0.2, tails are long)
    gate_sigma: float = 3.0
    gate_margin: float = 0.5          # m, box grown for the detections
    radar_cap: float = 4.0            # detections of one box count at most like this many independent ones
    v_stat: float = 0.5               # m/s  stationary rule
    d_stat: float = 1.0               # m
    ego_half_window: float = 0.09     # s, ego velocity from the poses t +- this
    conflict_share: float = 0.5       # radar_conflict when more than this share of >= 3 detections is rejected
    stat_moving_speed: float = 1.5    # m/s, median |Doppler| of a "stationary" box above which it is flagged


@dataclass
class ObjectFlow:
    id: str
    group: str
    xy: np.ndarray                    # world position at t
    yaw: float
    size: np.ndarray
    v: np.ndarray                     # (2,) world velocity m/s (NaN = unknown)
    cov: np.ndarray                   # (2, 2) m^2/s^2
    source: str                       # "stationary" | "track" | "track+radar" | "unknown"
    n_radar: int = 0                  # detections inside the box
    n_gated: int = 0                  # of which used
    flag: str = "ok"
    observed: bool = True
    stationary: bool = False

    @property
    def speed(self):
        return float(np.hypot(*self.v))


@dataclass
class FlowGT:
    t: float
    ego: tuple                        # (vx, vy, omega) in the ego frame at t
    objects: list = field(default_factory=list)
    flow_ego: np.ndarray = None       # (2, nx, ny) m/s, ego axes
    flow_obj: np.ndarray = None
    flow: np.ndarray = None
    weight: np.ndarray = None         # (nx, ny) 1 known / 0 unknown
    sigma: np.ndarray = None          # (nx, ny) m/s std of the speed estimate


def displace(flow, dt):
    """displacement (m) over any time dt (s): flow (m/s) * dt, first order."""
    return flow * dt


# ------------------------------------------------------------------------------------------------------ the function
def estimate_flow(tracks, ego, t, grid=None, radar=(), config=None):
    """see the module docstring; returns a FlowGT at time t."""
    cfg = config or Config()
    x0, y0, yaw0 = ego.pose(t)
    R = _rot(yaw0)                                                   # world <- ego
    ego_v = ego_velocity(ego, t, cfg.ego_half_window)               # step 1
    objs = []
    for tr in tracks:
        o = _object_at(tr, t, cfg)                                   # step 2
        if o is not None:
            objs.append(o)
    for o in objs:
        _radar_update(o, radar, tracks_by_id={tr.id: tr for tr in tracks}, cfg=cfg)   # steps 3, 4
    fg = FlowGT(t=t, ego=ego_v, objects=objs)
    if grid is not None:
        _rasterise(fg, grid, (x0, y0, yaw0))                          # step 5
    return fg


# ------------------------------------------------------------------------------------------------------- the helpers
def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def ego_velocity(ego, t, h=0.09):
    """(vx, vy, omega) in the ego frame at t [m/s, m/s, rad/s] from the poses at t -+ h (real time difference). Inside h of the first or
    last pose the window is moved to lie inside the poses (a one-sided difference over 2 h), never clamped: `Ego.pose` holds its end
    values outside the range, which halved the speed at the first and the last keyframe of a sequence."""
    lo, hi = t - h, t + h
    if lo < ego.t[0]:
        lo, hi = ego.t[0], min(ego.t[0] + 2 * h, ego.t[-1])
    elif hi > ego.t[-1]:
        lo, hi = max(ego.t[-1] - 2 * h, ego.t[0]), ego.t[-1]
    if hi <= lo:                                                    # fewer than two poses: no motion can be told
        return 0.0, 0.0, 0.0
    (xa, ya, wa), (xb, yb, wb) = ego.pose(lo), ego.pose(hi)
    v = _rot(ego.pose(t)[2]).T @ np.array([xb - xa, yb - ya]) / (hi - lo)
    return float(v[0]), float(v[1]), float((wb - wa) / (hi - lo))


def is_stationary(tr, cfg):
    if tr.stationary is not None:
        return tr.stationary
    if len(tr.t) < 3:
        return False
    sp = np.linalg.norm(tr.xy[2:] - tr.xy[:-2], axis=1) / (tr.t[2:] - tr.t[:-2])        # central differences, m/s
    spread = np.linalg.norm(tr.xy - np.median(tr.xy, 0), axis=1)
    return bool(sp.max() < cfg.v_stat and spread.max() < cfg.d_stat)


def _object_at(tr, t, cfg):
    """pose and velocity (with covariance) of one track at time t, or None when the track does not cover t (+- 0.3 s)."""
    if t < tr.t[0] - 0.3 or t > tr.t[-1] + 0.3:
        return None
    stat = is_stationary(tr, cfg)
    obs = bool(tr.observed[np.argmin(np.abs(tr.t - t))])
    if stat:
        xy = np.median(tr.xy, 0)
        yaw = math.atan2(np.sin(tr.yaw).mean(), np.cos(tr.yaw).mean())
        return ObjectFlow(tr.id, tr.group, xy, yaw, tr.size, np.zeros(2), np.eye(2) * 0.05 ** 2, "stationary",
                          flag="ok" if obs else "unobserved", observed=obs, stationary=True)
    xy = np.array([np.interp(t, tr.t, tr.xy[:, 0]), np.interp(t, tr.t, tr.xy[:, 1])])
    yaw = float(np.interp(t, tr.t, np.unwrap(tr.yaw)))
    idx = np.nonzero(np.abs(tr.t - t) <= cfg.window)[0]
    if len(idx) < 3:
        return ObjectFlow(tr.id, tr.group, xy, yaw, tr.size, np.full(2, np.nan), np.eye(2) * 1e4, "unknown",
                          flag="short_track", observed=obs)
    tau = tr.t[idx] - t
    A = np.stack([np.ones_like(tau), tau], 1)
    coef, *_ = np.linalg.lstsq(A, tr.xy[idx], rcond=None)
    var = cfg.sigma_pos ** 2 / max(float(((tau - tau.mean()) ** 2).sum()), 1e-6)
    return ObjectFlow(tr.id, tr.group, xy, yaw, tr.size, coef[1].copy(), np.eye(2) * var, "track",
                      flag="ok" if obs else "unobserved", observed=obs)


def _box_corners(xy, yaw, size):
    w, l = size[0], size[1]
    loc = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    return xy + loc @ _rot(yaw).T


def radar_points_in_box(o, tr, radar, margin):
    """(r_hat (K, 2), doppler (K,), sensor names) of the detections of all frames inside the box, moved to each frame's time
    by the track's own motion and grown by `margin` (m) on every side."""
    out_r, out_d, out_s = [], [], []
    for fr in radar:
        if len(fr.xy) == 0:
            continue
        if tr is not None and len(tr.t) >= 2:
            xy = np.array([np.interp(fr.t, tr.t, tr.xy[:, 0]), np.interp(fr.t, tr.t, tr.xy[:, 1])])
            if o.stationary:
                xy = o.xy
        else:
            xy = o.xy
        size = o.size + np.array([2 * margin, 2 * margin, 0.0])
        ins = _inside_quad(fr.xy, _box_corners(xy, o.yaw, size))
        if ins.any():
            r = fr.xy[ins] - fr.sensor_xy
            out_r.append(r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6))
            out_d.append(fr.doppler[ins])
            out_s.extend([fr.sensor] * int(ins.sum()))
    if not out_r:
        return np.zeros((0, 2)), np.zeros(0), []
    return np.vstack(out_r), np.concatenate(out_d), out_s


def _radar_update(o, radar, tracks_by_id, cfg):
    """steps 3 and 4: fuse the radial Doppler of the detections inside the box with the track velocity, set the flags."""
    if not radar:
        return
    rh, d, _ = radar_points_in_box(o, tracks_by_id[o.id], radar, cfg.gate_margin)
    o.n_radar = len(d)
    if len(d) == 0:
        return
    if o.stationary:
        if len(d) >= 3 and abs(float(np.median(d))) > cfg.stat_moving_speed:
            o.flag = "stationary_but_moving"
        return
    if o.source == "unknown":
        return
    v0, S0 = o.v.copy(), o.cov.copy()
    P0 = np.linalg.inv(S0)
    v = v0.copy()
    gate = np.ones(len(d), bool)
    for _ in range(2):
        resid = rh @ v - d
        sd = np.sqrt(cfg.sigma_doppler ** 2 + np.einsum("ij,jk,ik->i", rh, S0 if _ == 0 else o.cov, rh))
        gate = np.abs(resid) <= cfg.gate_sigma * sd
        if not gate.any():
            break
        n = int(gate.sum())
        w = 1.0 / (cfg.sigma_doppler ** 2 * max(1.0, n / cfg.radar_cap))
        Lam = P0 + w * rh[gate].T @ rh[gate]
        eta = P0 @ v0 + w * rh[gate].T @ d[gate]
        o.cov = np.linalg.inv(Lam)
        v = o.cov @ eta
    o.n_gated = int(gate.sum())
    if o.n_gated > 0:
        o.v, o.source = v, "track+radar"
    if len(d) >= 3 and (1 - o.n_gated / len(d)) > cfg.conflict_share:
        o.flag, o.v, o.cov, o.source = "radar_conflict", v0, S0, "track"
        cons = _radial_consensus(v0, rh, d, cfg)
        if cons is not None:                       # the detections agree with each other, not with the track: radar wins
            o.v, o.cov, o.source, o.flag = cons[0], cons[1], "radar", "radar_override"


def _radial_consensus(v0, rh, d, cfg):
    """Huber-robust speed s along the track's direction u (d_i = s * r_hat_i . u). Returns (v = s u, cov) when at least 3
    usable detections agree among themselves (robust spread < 0.7 m/s, >= 60 % within 1 m/s), else None."""
    sp = float(np.hypot(*v0))
    if sp < 0.5:
        return None
    u = v0 / sp
    a = rh @ u
    use = np.abs(a) >= 0.3
    if use.sum() < 3:
        return None
    a, dd = a[use], d[use]
    s = float(np.median(dd / a))
    for _ in range(6):
        r = dd - s * a
        w = np.minimum(1.0, 0.7 / np.maximum(np.abs(r), 1e-6))
        s = float((w * a * dd).sum() / (w * a * a).sum())
    r = dd - s * a
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    if mad >= 0.7 or np.mean(np.abs(r) < 1.0) < 0.6:
        return None
    var_s = (mad ** 2 + 0.3 ** 2) / float((w * a * a).sum())
    return s * u, var_s * np.outer(u, u) + 0.2 ** 2 * (np.eye(2) - np.outer(u, u))


def ego_flow_field(grid, ego_v):
    """apparent velocity (2, nx, ny) m/s of static world points in the ego frame: -(v + omega x r), ego_v = (vx, vy, omega)."""
    vx, vy, om = ego_v
    gx, gy = cell_centres(grid)
    return np.stack([-vx + om * gy, -vy - om * gx]).astype(np.float32)


def _rasterise(fg, grid, pose):
    x0, y0, yaw0 = pose
    gx, gy = cell_centres(grid)
    vx, vy, om = fg.ego
    fg.flow_ego = ego_flow_field(grid, fg.ego)
    fg.flow_obj = np.zeros_like(fg.flow_ego)
    fg.weight = np.ones(gx.shape, np.float32)
    fg.sigma = np.full(gx.shape, 0.1, np.float32)
    Rinv = _rot(yaw0).T
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    for o in fg.objects:
        quad = (_box_corners(o.xy, o.yaw, o.size) - np.array([x0, y0])) @ Rinv.T        # ego frame at t
        if quad[:, 0].max() < grid.x_min or quad[:, 0].min() > grid.x_max or quad[:, 1].max() < grid.y_min \
                or quad[:, 1].min() > grid.y_max:
            continue
        m = _inside_quad(P, quad).reshape(gx.shape)
        if not m.any():
            continue
        if not o.observed or o.source == "unknown":
            fg.weight[m] = 0.0
            continue
        ve = Rinv @ o.v
        fg.flow_obj[0][m], fg.flow_obj[1][m] = ve[0], ve[1]
        fg.sigma[m] = math.sqrt(float(np.trace(o.cov)) / 2)
    fg.flow = fg.flow_ego + fg.flow_obj


# ------------------------------------------------------------------------------------------------------- self-test
def _synthetic(seed=0, rate=2.0, sigma_pos=0.15, sigma_d=0.3, outlier=0.1):
    """a world with known truth: ego driving a gentle left curve, a car on a curved path, a parked car, a pedestrian;
    annotations sampled at `rate` Hz with position noise, radar detections with Doppler noise and outliers."""
    rng = np.random.default_rng(seed)
    tt = np.arange(0.0, 8.0 + 1e-9, 1.0 / rate)
    te = np.arange(0.0, 8.0 + 1e-9, 1.0 / 12)
    ego = Ego(te, 8.0 * te, 0.3 * np.sin(0.4 * te) * 8.0, 0.25 * np.sin(0.4 * te))      # a snake-ish path, speed ~ 8 m/s

    def car_pos(t):                                                                      # oncoming car on a curve
        return np.stack([60.0 - 9.0 * t, 3.5 + 0.4 * t + 0.05 * t ** 2], -1)

    def car_vel(t):
        return np.stack([np.full_like(t, -9.0), 0.4 + 0.1 * t], -1)

    ped_p0, ped_v = np.array([25.0, -6.0]), np.array([0.0, 1.4])
    tracks = [
        Track("car", tt, car_pos(tt) + rng.normal(0, sigma_pos, (len(tt), 2)), np.full(len(tt), math.pi), [1.9, 4.5, 1.6]),
        Track("parked", tt, np.array([35.0, -4.0]) + rng.normal(0, sigma_pos, (len(tt), 2)), np.zeros(len(tt)), [1.9, 4.5, 1.6]),
        Track("ped", tt, ped_p0 + ped_v * tt[:, None] + rng.normal(0, sigma_pos, (len(tt), 2)), np.zeros(len(tt)), [0.6, 0.6, 1.7],
              group="pedestrian"),
    ]
    truth = {"car": (car_pos, car_vel), "parked": (lambda t: np.tile([35.0, -4.0], (len(t), 1)), lambda t: np.zeros((len(t), 2))),
             "ped": (lambda t: ped_p0 + ped_v * t[:, None], lambda t: np.tile(ped_v, (len(t), 1)))}
    return rng, ego, tracks, truth, tt


def _synthetic_radar(rng, ego, truth, t, sigma_d=0.3, outlier=0.1, sensors=(("front", 0.0), ("left", 1.0), ("right", -1.0))):
    """radar frames at time t: two to four detections per box seen from three sensors at the ego position."""
    x, y, yaw = ego.pose(t)
    frames = []
    for name, side in sensors:
        pts, dop = [], []
        for tid, (pos_f, vel_f) in truth.items():
            p, v = pos_f(np.array([t]))[0], vel_f(np.array([t]))[0]
            sens = np.array([x, y]) + _rot(yaw) @ np.array([2.0, side * 0.8])
            for k in range(3):
                q = p + rng.normal(0, 0.3, 2)
                r = (q - sens) / np.linalg.norm(q - sens)
                d = float(r @ v) + rng.normal(0, sigma_d)
                if rng.random() < outlier:
                    d += rng.choice([-1, 1]) * rng.uniform(3, 8)
                pts.append(q)
                dop.append(d)
        frames.append(RadarFrame(t, name, np.array([x, y]) + _rot(yaw) @ np.array([2.0, side * 0.8]), np.array(pts), np.array(dop)))
    return frames


def _selftest():
    cfg = Config()
    grid = GridSpec()
    errs = {"track": [], "fused": []}
    along = {"track": [], "fused": []}
    for seed in range(30):
        rng, ego, tracks, truth, tt = _synthetic(seed)
        t = 3.0 + 0.5 * (seed % 5)
        radar = _synthetic_radar(rng, ego, truth, t)
        base = estimate_flow(tracks, ego, t, grid)
        fused = estimate_flow(tracks, ego, t, grid, radar)
        for name, fg in (("track", base), ("fused", fused)):
            for o in fg.objects:
                if o.id == "car":
                    vt = truth["car"][1](np.array([t]))[0]
                    errs[name].append(np.linalg.norm(o.v - vt))
                    along[name].append(abs((o.v - vt)[0]))                       # the radial direction of the oncoming car
    rm = {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in errs.items()}
    assert rm["fused"] < rm["track"], rm                                           # the radar correction helps
    # stationary rule and the pedestrian
    rng, ego, tracks, truth, tt = _synthetic(1)
    fg = estimate_flow(tracks, ego, 4.0, grid, _synthetic_radar(rng, ego, truth, 4.0))
    by = {o.id: o for o in fg.objects}
    assert by["parked"].stationary and np.hypot(*by["parked"].v) == 0
    assert np.linalg.norm(by["ped"].v - np.array([0.0, 1.4])) < 0.8 and not by["ped"].stationary, by["ped"].v
    # ego flow is exact for static points: -(v + omega x r) against the pose change over a short time
    vx, vy, om = fg.ego
    x1, y1, w1 = ego.pose(4.0)
    x2, y2, w2 = ego.pose(4.0 + 1 / 24)
    P = np.array([[30.0, 10.0]])                                                   # a static world point, ego frame at 4.0 s
    world = np.array([x1, y1]) + _rot(w1) @ P[0]
    moved = _rot(w2).T @ (world - np.array([x2, y2]))                               # the same point seen from the ego at 4.0 + dt
    ix, iy = int((30.0 - grid.x_min) / grid.res), int((10.0 - grid.y_min) / grid.res)
    gx, gy = cell_centres(grid)
    c = np.array([gx[ix, iy], gy[ix, iy]])
    world_c = np.array([x1, y1]) + _rot(w1) @ c
    pred = c + fg.flow_ego[:, ix, iy] * (1 / 24)
    exact = _rot(w2).T @ (world_c - np.array([x2, y2]))
    assert np.abs(pred - exact).max() < 0.02, (pred, exact)
    # time scaling: the same motion annotated at 2 Hz and at 4 Hz gives the same velocity (no dependence on the frame count)
    res = []
    for rate in (2.0, 4.0):
        rng, ego, tracks, truth, tt = _synthetic(7, rate=rate)
        o = {o.id: o for o in estimate_flow(tracks, ego, 4.0).objects}["car"]
        res.append(o.v)
    assert np.linalg.norm(res[0] - res[1]) < 0.8, res
    # an annotation with a gross error is flagged by the radar, and the fused velocity stays the track's
    rng, ego, tracks, truth, tt = _synthetic(3)
    tracks[0].xy = tracks[0].xy.copy()
    tracks[0].xy[:, 0] = (60.0 - 9.0 * 4.0) + 9.0 * (tt - 4.0)                     # right place at 4 s, wrong direction of travel
    fg = estimate_flow(tracks, ego, 4.0, grid, _synthetic_radar(rng, ego, truth, 4.0, outlier=0.0))
    car = {o.id: o for o in fg.objects}["car"]
    assert car.flag == "radar_override" and abs(car.v[0] - truth["car"][1](np.array([4.0]))[0, 0]) < 1.0, (car.flag, car.v)
    # unobserved boxes get weight 0 on their footprint, dense arrays have the right shapes
    tracks[1].observed[:] = False
    fg = estimate_flow(tracks, ego, 4.0, grid)
    assert fg.flow.shape == (2, grid.nx, grid.ny) and fg.weight.min() == 0 and fg.weight.max() == 1
    print(f"autogt flow selftest ok: car velocity RMSE track {rm['track']:.2f} -> with radar {rm['fused']:.2f} m/s "
          f"(30 trials, 3 sensors, 10 % outliers); radial error {np.mean(along['track']):.2f} -> {np.mean(along['fused']):.2f}; "
          f"parked car v = 0, pedestrian {by['ped'].v.round(2)} (truth 0, 1.4); ego flow vs exact pose within 2 cm per 1/24 s; "
          f"2 Hz and 4 Hz annotations agree; wrong direction of travel in the annotation found by the radar (radar_override, forward speed back within 1 m/s of the truth); unobserved box -> weight 0")


if __name__ == "__main__":
    _selftest()
