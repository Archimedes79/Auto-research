"""Self-test of the flow box (flow.py) on a synthetic world with known truth: `python -m GtGenerator selftest --only flow`.

The world: an ego driving a gentle left curve, an oncoming car on a curved path, a parked car and a pedestrian; annotations at 2 Hz with position noise, radar detections with
Doppler noise and outliers. Checked: the radar correction beats the track alone, the stationary rule, the pedestrian, the ego flow against the exact pose change, the same motion
at 2 Hz and at 4 Hz, a wrong direction of travel found by the radar (radar_override), an unobserved stationary box keeps its flag."""
import math

import numpy as np

from .flow import Config, Ego, RadarFrame, Track, _rot, ego_flow_field, estimate_flow
from .grid import GridSpec
from .layers import cell_centres


def _synthetic(seed=0, rate=2.0, sigma_pos=0.15):
    """a world with known truth: ego driving a gentle left curve, a car on a curved path, a parked car, a pedestrian;
    annotations sampled at `rate` Hz with position noise."""
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


def run(verbose=True):
    cfg = Config()
    assert cfg.v_stat == 0.5 and cfg.d_stat == 1.0, "the stationary rule of 2026-10-26"
    grid = GridSpec()
    errs = {"track": [], "fused": []}
    along = {"track": [], "fused": []}
    for seed in range(30):
        rng, ego, tracks, truth, tt = _synthetic(seed)
        t = 3.0 + 0.5 * (seed % 5)
        radar = _synthetic_radar(rng, ego, truth, t)
        base = estimate_flow(tracks, ego, t)
        fused = estimate_flow(tracks, ego, t, radar)
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
    fg = estimate_flow(tracks, ego, 4.0, _synthetic_radar(rng, ego, truth, 4.0))
    by = {o.id: o for o in fg.objects}
    assert by["parked"].stationary and np.hypot(*by["parked"].v) == 0
    assert np.linalg.norm(by["ped"].v - np.array([0.0, 1.4])) < 0.8 and not by["ped"].stationary, by["ped"].v
    # ego flow is exact for static points: -(v + omega x r) against the pose change over a short time
    x1, y1, w1 = ego.pose(4.0)
    x2, y2, w2 = ego.pose(4.0 + 1 / 24)
    flow_ego = ego_flow_field(grid, fg.ego)
    assert flow_ego.shape == (2, grid.nx, grid.ny) and flow_ego.dtype == np.float32
    ix, iy = int((30.0 - grid.x_min) / grid.res), int((10.0 - grid.y_min) / grid.res)
    gx, gy = cell_centres(grid)
    c = np.array([gx[ix, iy], gy[ix, iy]])
    world_c = np.array([x1, y1]) + _rot(w1) @ c
    pred = c + flow_ego[:, ix, iy] * (1 / 24)
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
    fg = estimate_flow(tracks, ego, 4.0, _synthetic_radar(rng, ego, truth, 4.0, outlier=0.0))
    car = {o.id: o for o in fg.objects}["car"]
    assert car.flag == "radar_override" and abs(car.v[0] - truth["car"][1](np.array([4.0]))[0, 0]) < 1.0, (car.flag, car.v)
    # an unobserved stationary box keeps its flag (it is what the GT puts into vel_flag), the others stay "ok"
    tracks[1].observed[:] = False
    flags = {o.id: o.flag for o in estimate_flow(tracks, ego, 4.0).objects}
    assert flags["parked"] == "unobserved" and flags["ped"] == "ok", flags
    msg = (f"flow selftest ok: car velocity RMSE track {rm['track']:.2f} -> with radar {rm['fused']:.2f} m/s (30 trials, 3 sensors, 10 % outliers); radial error "
           f"{np.mean(along['track']):.2f} -> {np.mean(along['fused']):.2f}; parked car v = 0, pedestrian {by['ped'].v.round(2)} (truth 0, 1.4); ego flow vs exact pose within 2 cm per "
           f"1/24 s; 2 Hz and 4 Hz annotations agree; wrong direction of travel found by the radar (radar_override); unobserved stationary box keeps its flag")
    if verbose:
        print(msg)
    return msg


if __name__ == "__main__":
    run()
