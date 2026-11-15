"""The flow box of GtGenerator: velocity ground truth of the tracked boxes from their annotations, the ego poses and (optionally) radar Doppler.

ONE function does the work:

    flow = estimate_flow(tracks, ego, t, radar=(), config=Config())

Everything is in SI units with REAL time stamps in seconds - velocities are m/s, never "per frame". No dataset knowledge lives here: a dataset adapter (e.g. `nuscenes_adapter.py`)
only turns its records into the small inputs below; the dense layers on the BEV grid are made by `nuscenes_gt.py` from the objects this returns (and `ego_flow_field`). numpy (and the cv2 that layers.py imports).

Inputs
  tracks  list[Track]      boxes over time: id, t (s, increasing), xy (world, m), yaw, size (w, l, h), group, observed
  ego     Ego              planar ego poses over time: t (s), x, y, yaw (world <- ego), any rate
  t       float            the frame time (s) the velocities are wanted at
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
Output FlowGT: the ego motion (vx, vy, omega) and `objects` (list[ObjectFlow]: pose at t, v_world, cov, source, n_radar, flag). The apparent velocity of a static world point caused by the
ego motion is `ego_flow_field(grid, ego)`: flow_ego = -(v_ego + omega x r); the flow of a cell is flow_ego + the velocity of the box that covers it (nuscenes_gt.py, first order).
Self-test (synthetic world with known truth): `python -m GtGenerator selftest --only flow` (flow_selftest.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

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
    source: str                       # "stationary" | "track" | "track+radar" | "radar" | "unknown"
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


# ------------------------------------------------------------------------------------------------------ the function
def estimate_flow(tracks, ego, t, radar=(), config=None):
    """see the module docstring; returns a FlowGT at time t."""
    cfg = config or Config()
    ego_v = ego_velocity(ego, t, cfg.ego_half_window)               # step 1
    objs = []
    for tr in tracks:
        o = _object_at(tr, t, cfg)                                   # step 2
        if o is not None:
            objs.append(o)
    for o in objs:
        _radar_update(o, radar, tracks_by_id={tr.id: tr for tr in tracks}, cfg=cfg)   # steps 3, 4
    return FlowGT(t=t, ego=ego_v, objects=objs)


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
        cons = _radial_consensus(v0, rh, d)
        if cons is not None:                       # the detections agree with each other, not with the track: radar wins
            o.v, o.cov, o.source, o.flag = cons[0], cons[1], "radar", "radar_override"


def _radial_consensus(v0, rh, d):
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
