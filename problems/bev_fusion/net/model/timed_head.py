"""Time decoding inside EVERY head (the owner, 2026-10-26): a head predicts at any delta t, plus an explicit velocity head.

    state S(t_c) (B, C, X, Y) from the trunk, dt, ego speed v and yaw rate omega
      -> MotionHead: velocity field v_hat (B, 2, X, Y), m/s in the ego axes at t_c    (an explicit, supervised output)
      -> TimeMover (shared by all heads): splat of S to t_c + dt with ego motion and v_hat, mass channel (0 = unknown cell)
      -> per head: refine block on [S(t_c + dt); mass] with FiLM by (dt, |dt|, sign dt), residual, zero start
      -> the head itself (semantic, object, sensor auxiliary, ...) on the refined state

The motion head is the same quantity the splat uses (plan 7.1) and the object head regresses per object centre (`vel`): one
velocity field, one move of the state, any number of heads that decode it at t_c + dt. At dt = 0 with the zero-started refine
block the wrapper reproduces the plain head exactly. Heads are any modules mapping (B, C, X, Y) -> tensor or dict.
Tests: tests/test_timed_head.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bev_fusion.net.model.lift_pd import Grid
from bev_fusion.net.model.splat import VelocityHead, splat_bev


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


class TimeMover(nn.Module):
    """Move the state to t_c + dt (either sign): ego motion + velocity head + bilinear splat with a mass channel.
    Shared by all heads. Returns (state at t_c + dt, mass (B, 1, X, Y), v_hat), with `carry` a 4th output (the carried channels).
    With a `carry` the splat runs over [state; carry]: the first C channels are the state moved exactly as a call without carry would move it (every channel
    is scattered on its own and the weights do not depend on the channels)."""

    def __init__(self, c, grid=None, hidden=64, v_max=40.0):
        super().__init__()
        self.grid = grid or Grid()
        self.motion = MotionHead(c, hidden, v_max)

    def forward(self, state, dt, v, omega, v_field=None, carry=None):
        """v_field (B, 2, X, Y): a velocity computed elsewhere (the motion block, ModelConfig.motion_block) used instead of
        self.motion; carry (B, K, X, Y): extra channels moved with the state (the field itself, for the box velocity at dt),
        returned as a 4th output."""
        if v_field is None and self.motion is None:
            raise ValueError("TimeMover without its velocity head (ModelConfig.motion_block) needs v_field on every call")
        if carry is None:
            head = self.motion if v_field is None else (lambda _s: v_field)
            return splat_bev(state, self.grid, dt, v, omega, head=head)
        vf = self.motion(state) if v_field is None else v_field   # the head reads the state only, never the carried channels
        head = lambda _s: vf  # noqa: E731
        C = state.shape[1]
        out, mass, v_hat = splat_bev(torch.cat([state, carry.to(state.dtype)], 1), self.grid, dt, v, omega, head=head)
        return out[:, :C], mass, v_hat, out[:, C:]


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

    def forward(self, state, dt=0.0, v=0.0, omega=0.0, return_aux=False, v_field=None, carry=None):
        B = state.shape[0]
        if carry is None:
            moved, mass, v_hat = self.mover(state, dt, v, omega, v_field=v_field)
            carried = None
        elif not return_aux:
            raise ValueError("carry is returned in the aux dict: call with return_aux=True")
        else:
            moved, mass, v_hat, carried = self.mover(state, dt, v, omega, v_field=v_field, carry=carry)
        gamma, beta = self.film(dt_embedding(dt, B, state)).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        h = self.refine(torch.cat([moved, mass], 1)) * (1 + gamma) + beta
        x = moved + self.out(F.relu(h))
        y = self.head(x)
        return (y, {"mass": mass, "v_hat": v_hat, "state": x, "carry": carried}) if return_aux else y
