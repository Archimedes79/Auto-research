"""Is the map-image offset a TIME offset? (Mirko, 2026-10-26)

A time offset tau between the image and the ego pose used for the GT moves the map along the driving direction by
about v tau and turns it by omega tau - it scales with speed (no effect when standing); a map / localisation offset
does not. Per keyframe of nuScenes mini (dividers at 6-20 m, exact map polylines):
1. camera time minus annotation (sample = lidar) time
2. contrast score (paint minus 0.4 m to either side, bilinear image sampling) over a grid of
   tau in -0.3..+0.3 s (ego pose at t_cam + tau, interpolated along the 12 Hz camera track) x lateral shift dy in
   -1.5..+1.5 m; best tau at dy = 0, best dy at tau = 0, best joint; how much each explains (score gain over (0, 0))
3. is the best tau related to the speed? (timing: implied longitudinal shift v tau consistent across frames;
   standing frames: tau has no effect)
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_gt_time_offset.py
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from gt_bev import NuScenesGT  # noqa: E402
from lift_pd import Grid  # noqa: E402
from view_training_samples import project  # noqa: E402

G = Grid()
TAUS = np.round(np.arange(-0.3, 0.301, 0.025), 3)
DYS = np.round(np.arange(-1.5, 1.501, 0.125), 3)
root = os.path.expanduser("~/data/nuscenes")
src = NuScenesGT(root, "v1.0-mini")


def dense(polylines, step=0.05):
    """points every `step` metres along the polylines (copied from check_gt_vector_vs_cells.py, which runs at import)."""
    out = []
    for pl in polylines:
        for a, b in zip(pl[:-1], pl[1:]):
            n = max(1, int(np.hypot(*(b - a)) / step))
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            out.append(a + t * (b - a))
    return np.concatenate(out) if out else np.zeros((0, 2))


def bilinear(img, uv):
    H, W = img.shape
    u, v = uv[:, 0], uv[:, 1]
    ok = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    u0, v0 = np.floor(u[ok]).astype(int), np.floor(v[ok]).astype(int)
    fu, fv = u[ok] - u0, v[ok] - v0
    val = np.full(len(u), np.nan)
    val[ok] = (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
               + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)
    return val


def pose_at(track_t, track_T, t):
    """ego pose (4x4) at time t (us) by linear interpolation of translation and yaw along the camera track."""
    i = int(np.clip(np.searchsorted(track_t, t) - 1, 0, len(track_t) - 2))
    a = (t - track_t[i]) / (track_t[i + 1] - track_t[i])
    Ta, Tb = track_T[i], track_T[i + 1]
    ya, yb = math.atan2(Ta[1, 0], Ta[0, 0]), math.atan2(Tb[1, 0], Tb[0, 0])
    y = ya + a * ((yb - ya + math.pi) % (2 * math.pi) - math.pi)
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    T[:3, 3] = (1 - a) * Ta[:3, 3] + a * Tb[:3, 3]
    return T


def score(gt, img, xy):
    uv, z = project(gt, np.c_[xy, np.zeros(len(xy))])
    v = bilinear(img, uv)
    v[z <= 1] = np.nan
    return v


rows = []
for sc in src.nusc.scene:
    tok = sc["first_sample_token"]
    for k in range(40):
        s = src.nusc.get("sample", tok)
        if k % 8 == 4:
            gt = src.keyframe_gt(tok, G)
            vec = dense(gt["meta"]["vectors"]["lane_divider"] + gt["meta"]["vectors"]["road_divider"])
            if len(vec):
                vec = vec[(vec[:, 0] > 6) & (vec[:, 0] < 20)]
            if len(vec) > 50:
                img = np.asarray(Image.open(os.path.join(root, gt["meta"]["cam_file"])).convert("L"), np.float32)
                track = src.camera_track(tok)
                tt = np.array([x["timestamp"] for x in track], float)
                TT = [src.ego_pose(x) for x in track]
                t_cam = gt["meta"]["timestamp"]
                T0 = gt["meta"]["T_e2g"]
                Pg = np.c_[vec, np.zeros(len(vec)), np.ones(len(vec))] @ T0.T          # world points of the lines
                S = np.full((len(TAUS), len(DYS)), np.nan)
                for i, tau in enumerate(TAUS):
                    Tt = pose_at(tt, TT, t_cam + tau * 1e6)
                    pe = (Pg @ np.linalg.inv(Tt).T)[:, :2]                            # lines in the ego frame at t+tau
                    for j, dy in enumerate(DYS):
                        c = score(gt, img, pe + [0, dy]) - 0.5 * (score(gt, img, pe + [0, dy - 0.4])
                                                                  + score(gt, img, pe + [0, dy + 0.4]))
                        S[i, j] = np.nanmean(c) if np.isfinite(c).sum() > 30 else np.nan
                if np.isfinite(S).all():
                    i0, j0 = list(TAUS).index(0.0), list(DYS).index(0.0)
                    bi, bj = np.unravel_index(np.argmax(S), S.shape)
                    rel = np.linalg.inv(TT[max(0, np.searchsorted(tt, t_cam) - 1)]) @ TT[min(len(TT) - 1, np.searchsorted(tt, t_cam) + 1)]
                    dts = (tt[min(len(TT) - 1, np.searchsorted(tt, t_cam) + 1)] - tt[max(0, np.searchsorted(tt, t_cam) - 1)]) * 1e-6
                    v = abs(rel[0, 3]) / dts
                    rows.append({"scene": sc["name"], "k": k, "v": v, "cam_minus_sample_ms": (t_cam - s["timestamp"]) / 1e3,
                                 "s00": S[i0, j0], "tau_only": TAUS[int(np.argmax(S[:, j0]))], "gain_tau": S[:, j0].max() - S[i0, j0],
                                 "dy_only": DYS[int(np.argmax(S[i0]))], "gain_dy": S[i0].max() - S[i0, j0],
                                 "tau_joint": TAUS[bi], "dy_joint": DYS[bj], "gain_joint": S.max() - S[i0, j0]})
        if not s["next"]:
            break
        tok = s["next"]

print("scene      kf | v m/s | cam-sample ms | best tau alone (gain) | best dy alone (gain) | joint tau, dy (gain)")
for r in rows:
    print(f"{r['scene']} {r['k']:2d} | {r['v']:5.1f} | {r['cam_minus_sample_ms']:+7.1f} | {r['tau_only']:+.3f} s ({r['gain_tau']:5.1f}) | "
          f"{r['dy_only']:+.3f} m ({r['gain_dy']:5.1f}) | {r['tau_joint']:+.3f} s, {r['dy_joint']:+.3f} m ({r['gain_joint']:5.1f})")
R = rows
moving = [r for r in R if r["v"] > 2]
print(f"\nframes {len(R)} (moving > 2 m/s: {len(moving)}); camera minus annotation time: median "
      f"{np.median([r['cam_minus_sample_ms'] for r in R]):+.1f} ms, range {min(r['cam_minus_sample_ms'] for r in R):+.1f} .. "
      f"{max(r['cam_minus_sample_ms'] for r in R):+.1f} ms")
print(f"mean score gain: tau alone {np.mean([r['gain_tau'] for r in R]):.2f}, dy alone {np.mean([r['gain_dy'] for r in R]):.2f}, "
      f"joint {np.mean([r['gain_joint'] for r in R]):.2f}")
print(f"moving frames: best tau alone median {np.median([r['tau_only'] for r in moving]):+.3f} s, spread (IQR) "
      f"{np.subtract(*np.percentile([r['tau_only'] for r in moving], [75, 25])):.3f} s; implied v*tau median "
      f"{np.median([r['v'] * r['tau_only'] for r in moving]):+.2f} m")
print(f"best dy alone: median {np.median([r['dy_only'] for r in R]):+.3f} m, |dy| <= 0.25 m in {np.mean([abs(r['dy_only']) <= 0.25 for r in R]):.0%}")
