"""Time decoding inside EVERY head (Mirko, 2026-10-26): a head predicts at any delta t, plus an explicit velocity head.

    state S(t_c) (B, C, X, Y) from the trunk, dt, ego speed v and yaw rate omega
      -> MotionHead: velocity field v_hat (B, 2, X, Y), m/s in the ego axes at t_c    (an explicit, supervised output)
      -> TimeMover (shared by all heads): splat of S to t_c + dt with ego motion and v_hat, mass channel (0 = unknown cell)
      -> per head: refine block on [S(t_c + dt); mass] with FiLM by (dt, |dt|, sign dt), residual, zero start
      -> the head itself (semantic, object, sensor auxiliary, ...) on the refined state

The motion head is the same quantity the splat uses (plan 7.1) and the object head regresses per object centre (`vel`): one
velocity field, one move of the state, any number of heads that decode it at t_c + dt. At dt = 0 with the zero-started refine
block the wrapper reproduces the plain head exactly. Heads are any modules mapping (B, C, X, Y) -> tensor or dict.
Run the self-test: python training/timed_head.py
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .lift_pd import Grid
    from .splat import VelocityHead, splat_bev
except ImportError:                                   # run as a script
    from lift_pd import Grid
    from splat import VelocityHead, splat_bev


def dt_embedding(dt, B, like):
    """dt scalar or (B,) -> (B, 3): dt, |dt|, sign(dt)."""
    d = torch.as_tensor(dt, dtype=like.dtype, device=like.device).reshape(-1).expand(B)
    return torch.stack([d, d.abs(), torch.sign(d)], 1)


class MotionHead(nn.Module):
    """Explicit velocity head: state -> object velocity field (B, 2, X, Y) in m/s, ego axes at the state's time.
    Zero start (v_hat = 0: pure ego compensation), bounded by v_max."""

    def __init__(self, cin, hidden=64, v_max=40.0):
        super().__init__()
        self.f = VelocityHead(cin, hidden, v_max)

    def forward(self, state):
        return self.f(state)

    @staticmethod
    def loss(v_hat, v_gt, object_mask, min_speed=1.0, w_background=0.1):
        """L1 to the GT velocity on object cells (GT set to 0 below min_speed: a parked car does not move) plus a weak L1 to 0
        on the other cells (the background is static). v_hat, v_gt (B, 2, X, Y); object_mask (B, 1, X, Y) bool / {0, 1}."""
        m = object_mask.to(v_hat.dtype)
        v_gt = torch.where(v_gt.norm(dim=1, keepdim=True) < min_speed, torch.zeros_like(v_gt), v_gt)
        obj = ((v_hat - v_gt).abs() * m).sum() / (m.sum() * 2).clamp(min=1)
        bg = (v_hat.abs() * (1 - m)).sum() / ((1 - m).sum() * 2).clamp(min=1)
        return obj + w_background * bg


class TimeMover(nn.Module):
    """Move the state to t_c + dt (either sign): ego motion + velocity head + bilinear splat with a mass channel.
    Shared by all heads. Returns (state at t_c + dt, mass (B, 1, X, Y), v_hat)."""

    def __init__(self, c, grid=None, hidden=64, v_max=40.0, corr=None):
        super().__init__()
        self.grid, self.corr = grid or Grid(), corr
        self.motion = MotionHead(c, hidden, v_max)

    def forward(self, state, dt, v, omega):
        out, mass, v_hat = splat_bev(state, self.grid, dt, v, omega, head=self.motion, corr=self.corr)
        return out, mass, v_hat


class TimedHead(nn.Module):
    """A head that decodes at any dt: refine(S(t_c + dt), mass, dt) then `head`. `mover` is shared between the heads."""

    def __init__(self, head, c, mover):
        super().__init__()
        self.head, self.mover = head, mover
        self.refine = nn.Sequential(nn.Conv2d(c + 1, c, 3, padding=1), nn.ReLU())
        self.film = nn.Linear(3, 2 * c)
        self.out = nn.Conv2d(c, c, 3, padding=1)
        for m in (self.film, self.out):                # zero start: the wrapper is the identity on the state
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state, dt=0.0, v=0.0, omega=0.0, return_aux=False):
        B = state.shape[0]
        moved, mass, v_hat = self.mover(state, dt, v, omega)
        gamma, beta = self.film(dt_embedding(dt, B, state)).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        h = self.refine(torch.cat([moved, mass], 1)) * (1 + gamma) + beta
        x = moved + self.out(F.relu(h))
        y = self.head(x)
        return (y, {"mass": mass, "v_hat": v_hat, "state": x}) if return_aux else y


# ------------------------------------------------------------------------------------------------ self-test
def _selftest():
    torch.manual_seed(0)
    grid, C = Grid(), 16
    X, Y = grid.nx, grid.ny
    xs = grid.x_min + (torch.arange(X) + 0.5) * grid.res
    ys = grid.y_min + (torch.arange(Y) + 0.5) * grid.res

    def blob(x0, y0, s=1.5):
        b = torch.exp(-((xs[:, None] - x0) ** 2 + (ys[None, :] - y0) ** 2) / (2 * s * s))
        S = torch.zeros(1, C, X, Y)
        S[0, 0] = b
        return S

    def centroid_x(m):
        return float((m[0, 0] * xs[:, None]).sum() / m[0, 0].sum())

    pick = nn.Conv2d(C, 1, 1, bias=False)              # a head that reads channel 0
    with torch.no_grad():
        pick.weight.zero_()
        pick.weight[0, 0, 0, 0] = 1.0
    mover = TimeMover(C, grid)
    th = TimedHead(pick, C, mover)
    S = blob(30.0, 0.0)
    # 1 dt = 0: exactly the plain head (zero-started refine, identity splat), no unknown cells
    y, aux = th(S, 0.0, 10.0, 0.0, return_aux=True)
    assert torch.allclose(y, pick(S), atol=1e-6), "dt = 0 must be the plain head"
    assert float(aux["mass"].min()) > 0.99, "dt = 0: all cells known"
    # 2 ego motion: static content moves back by v dt (the car drives toward it); cells beyond the old far edge are unknown
    for dt, x_exp in ((0.5, 25.0), (-0.5, 35.0)):
        y, aux = th(S, dt, 10.0, 0.0, return_aux=True)
        assert abs(centroid_x(y) - x_exp) < 0.1, (dt, centroid_x(y))
    unk = lambda dt: float((th(S, dt, 10.0, 0.0, return_aux=True)[1]["mass"] < 0.5).float().mean())
    u5, u10 = unk(0.5), unk(1.0)
    assert 0.07 < u5 < 0.10 and 0.15 < u10 < 0.19 and u10 > u5, (u5, u10)       # about v dt / 60 m of the map is unknown
    # 3 object velocity: set the velocity head by hand to +4 m/s: the blob lands at x + v_obj dt (ego standing still)
    with torch.no_grad():
        mover.motion.f.net[2].bias.copy_(torch.tensor([40.0 * math.atanh(0.1), 0.0]))      # v_hat = 4.0 m/s along x
    y, aux = th(S, 0.5, 0.0, 0.0, return_aux=True)
    assert abs(float(aux["v_hat"][0, 0].mean()) - 4.0) < 1e-3
    assert abs(centroid_x(y) - 32.0) < 0.1, centroid_x(y)
    with torch.no_grad():
        mover.motion.f.net[2].bias.zero_()
    # 4 the motion loss: zero for the right velocity, background pulled to rest, parked objects (< 1 m/s) count as 0
    v_gt = torch.zeros(1, 2, X, Y)
    v_gt[0, 0, 40:50, 50:55] = 6.0
    mask = torch.zeros(1, 1, X, Y)
    mask[0, 0, 38:52, 48:57] = 1.0
    assert float(MotionHead.loss(v_gt.clone(), v_gt, mask)) < 1e-6
    assert float(MotionHead.loss(torch.zeros_like(v_gt), v_gt, mask)) > 0.5
    # 5 several heads share ONE mover; gradients reach the film, the refine block and the motion head
    from bev_trunk import SemanticHead
    from object_head import ObjectHead
    sem, obj = TimedHead(SemanticHead(C, 5), C, mover), TimedHead(ObjectHead(C), C, mover)
    assert sem.mover is obj.mover and sem.mover.motion.f.net[0].weight is obj.mover.motion.f.net[0].weight
    S2 = torch.randn(2, C, X, Y)
    ys = sem(S2, 0.5, 8.0, 0.1)
    yo = obj(S2, [0.5, -0.5], [8.0, 6.0], [0.1, -0.1])
    assert ys.shape == (2, 5, X, Y) and set(yo) == {"hm", "off", "size", "rot", "rad", "vel"}
    mover.zero_grad()
    for p in list(sem.parameters()) + list(obj.parameters()):
        p.grad = None
    (ys.mean() + yo["hm"].mean()).backward()
    assert sem.film.weight.grad is not None and obj.out.weight.grad is not None
    assert mover.motion.f.net[2].weight.grad is not None, "the motion head gets a gradient through the splat"
    n_shared = len({id(p) for p in list(sem.parameters()) + list(obj.parameters())})
    n_sum = len(list(sem.parameters())) + len(list(obj.parameters()))
    assert n_shared < n_sum, "the mover's parameters are shared, not duplicated"
    print(f"timed_head selftest ok: dt = 0 is the plain head, ego motion lands at x -+ v dt, unknown share {u5:.3f} / {u10:.3f} "
          f"at dt 0.5 / 1.0 s, object velocity lands at x + v dt, motion loss, two heads share one mover, gradients")


if __name__ == "__main__":
    _selftest()
