"""Does the Doppler write mask of the radar memory keep a moving object out of it? (card t19_radar_memory, the gate before any GPU time)

The radar memory (`temporal_fusion.TemporalFusion`, `ModelConfig.rad_static_gate`) writes with the weight  min(mass, 1) * static * obs_mass.
With rad_static_gate the obs_mass is `precompute.radar_static`: per cell the mean of exp(-(vr / 1 m/s)^2) of its detections, vr the
ego-compensated Doppler. The learned `static` map (velocity head) starts at 1.0 everywhere and the radar branch gives it no gradient, so
the Doppler mask is the ONLY thing that keeps a mover out. Synthetic world of the temporal_fusion self-test (static pattern, radar at the
origin, ego standing still, a dense radar: a detection in every cell), a blob of 6 m/s, memory read behind the blob compared with the static world:
  no gate         everything static (obs_mass = field of view only)             -> a ghost trail (> 0.1)
  radial mover    the blob drives along the line of sight (vr = -6 m/s)         -> no ghost (< 0.02): the Doppler sees it
  crossing mover  the blob drives across the line of sight at x = 16 m          -> a ghost where the radial speed is about 0 (the Doppler
                  of a crossing object vanishes at the crossing point): the documented limit of a Doppler-only gate, which a learned velocity
                  or the vectors of several radars have to cover. The test pins the size of this ghost so a change shows.
Run from the bev_fusion folder: wsl ~/miniconda3/envs/gcar/bin/python tests/test_radar_memory_ghost.py
"""
import sys

import torch

sys.path.insert(0, "training")
from lift_pd import Grid  # noqa: E402
from splat import grid_points  # noqa: E402
from temporal_fusion import TemporalFusion, _observe, pattern  # noqa: E402

DT, FRAMES, X_FOV, SIGMA = 1 / 12, 30, 8.0, 1.0
fails = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def drive(mover, gate):
    """-> max ghost over frames 10-29: mean |memory read - static world| on the cells behind the mover (axis of its path)."""
    torch.manual_seed(0)
    grid = Grid(x_min=0.0, x_max=24.0, y_min=-8.0, y_max=8.0, res=0.5)
    X, Y = grid.nx, grid.ny
    mod = TemporalFusion(3, 8, grid, mem_cells=(128, 96), w_max=8.0).eval()          # velocity head at its zero start: static = 1
    pts = grid_points(grid)
    state, ghosts = None, []
    for t in range(FRAMES):
        pos, vel = mover(t * DT)
        blob = lambda w: torch.exp(-(((w[..., 0] - pos[0]) ** 2 + (w[..., 1] - pos[1]) ** 2) / 0.8))
        cur, w, om = _observe(torch.zeros(1, 3), pts, X_FOV, blob)
        on = blob(w) > 0.02
        speed = (vel[0] * w[..., 0] + vel[1] * w[..., 1]) / w.norm(dim=-1).clamp(min=1e-3)       # radial speed of the blob's cells
        vr = torch.where(on, speed, torch.zeros_like(speed))
        mask = om * (torch.exp(-(vr / SIGMA) ** 2) if gate else torch.ones_like(vr))             # precompute.radar_static x field of view
        cur3 = torch.stack([cur[:, 0], torch.zeros_like(cur[:, 0]), torch.zeros_like(cur[:, 0])], 1).view(1, -1, X, Y)
        with torch.no_grad():
            _, state, aux = mod(cur3, (), state, None, mask.view(1, 1, X, Y))
        conf = aux["mem_conf"][:, 0] / mod.w_max
        read = aux["mem_read"][:, 0]
        truth = pattern(w[..., 0], w[..., 1]).view(1, X, Y)
        d = torch.stack([w[..., 0] - pos[0], w[..., 1] - pos[1]], -1).view(1, X, Y, 2)                  # cell - blob centre
        v = torch.tensor(vel) / max(float(torch.tensor(vel).norm()), 1e-6)
        along = d[..., 0] * v[0] + d[..., 1] * v[1]
        across = d[..., 0] * -v[1] + d[..., 1] * v[0]
        path = (across.abs() < 1.2) & (along < -2.0) & (conf > 0.05)                                    # behind the blob, on its path
        ghosts.append((read - truth)[path].abs().mean().item() if path.any() else 0.0)
    return max(ghosts[10:])


crossing = lambda s: ((16.0, -6.0 + 6.0 * s), (0.0, 6.0))            # across the line of sight of a radar at the origin
radial = lambda s: ((22.0 - 6.0 * s, 2.0), (-6.0, 0.0))              # towards the radar (y = 2 m: no division by a vanishing radius)
g_none = drive(crossing, gate=False)
g_rad = drive(radial, gate=True)
g_cross = drive(crossing, gate=True)
print(f"ghost (mean deviation from the static world behind the blob, max over frames 10-29): no gate {g_none:.3f}, radial mover with the "
      f"Doppler gate {g_rad:.3f}, crossing mover with the Doppler gate {g_cross:.3f}")
check(g_none > 0.1, f"without a gate the mover leaves a ghost trail ({g_none:.3f} > 0.1)")
check(g_rad < 0.02, f"a mover on the line of sight is kept out by the Doppler gate ({g_rad:.3f} < 0.02)")
check(g_cross < g_none and g_cross > 3 * g_rad, f"a crossing mover is only partly kept out: {g_cross:.3f} (no gate {g_none:.3f}, radial {g_rad:.3f}): "
      "the Doppler is about 0 at the crossing point - the limit of a Doppler-only gate")
print("RADAR MEMORY GHOST GATE: " + ("PASS" if not fails else f"FAIL ({len(fails)})"))
sys.exit(1 if fails else 0)
