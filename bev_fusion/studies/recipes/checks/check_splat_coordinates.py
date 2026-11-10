"""Chained temporal splat with coordinates as appended channels (Mirko, 2026-10-25) against chained map splats.

Idea tested: where a stage needs positions (the splat), append an x, y map to the feature map, splat features and
coordinates together, use the splatted coordinates as the exact source positions of the next step, and strip x, y again
so that no learned layer ever sees absolute positions (no position overfitting). Same synthetic scenes and exact
constant-turn-rate expectation as check_splat_scenes.py (objects with their own velocity, lane lines, ego motion).

Variants (all with the same velocity head reading the feature channels vx, vy):
  A  chained map splats                      positions = cell centres at every step (the previous check)
  B  A + mass gate                           features multiplied by min(mass, 1) after every splat (tail cells that
                                             received little mass no longer re-appear at full strength)
  C  coordinates carried (x, y appended)     source positions of the next step = splatted coordinates; no gate
  D  C + mass gate                           coordinates carried and the gate
  E  one splat from the original positions   the whole time in one step (reference; what a chain should reach)
Measured per case: centre error of the four objects (m), mean IoU of the three cars at 0.5, lane lines' mean lateral
error (m). Run from bev_fusion: wsl ~/miniconda3/envs/gcar/bin/python temp/checks/check_splat_coordinates.py
"""
import sys

import torch

sys.path.insert(0, "training")
sys.path.insert(0, "temp/checks")
from check_splat_scenes import (DT, G, expected_scene, measure, pass_through_head,  # noqa: E402
                                source_scene)
from splat import grid_points, splat, splat_positions  # noqa: E402

THR = 0.05                      # mass below this: the cell holds no content (its stored coordinate is not used)


def chain_variant(state, steps, dt, v, w, head, coords, gate, imp_scale=6.0):
    B, C, X, Y = state.shape
    centres = grid_points(G, dtype=DT).unsqueeze(0)
    P = centres.clone()                                   # the x, y channels (source positions of the content)
    mass = None
    for _ in range(steps):
        v_obj = head(state).permute(0, 2, 3, 1).reshape(B, X * Y, 2)      # the head sees features only
        src = P if coords else centres
        pos_t = splat_positions(src, dt / steps, v, w, v_obj)
        feats = state.reshape(B, C, X * Y)
        if coords:
            feats = torch.cat([feats, pos_t.permute(0, 2, 1)], 1)         # append x, y, splat them together
        out, mass = splat(feats, pos_t, G, imp_scale * state[:, 0].reshape(B, X * Y))
        new = out[:, :C]
        if coords:
            has = (mass.reshape(B, X * Y, 1) > THR)
            P = torch.where(has, out[:, C:].reshape(B, 2, X * Y).permute(0, 2, 1), centres)   # x, y: side channels ...
        if gate:
            new = new * mass.clamp(max=1.0)
        state = new.reshape(B, C, X, Y)                                    # ... and strip them from the features
    return state, mass


def run(dt, v, w, steps, variant):
    src, lane_pts = source_scene()
    head = pass_through_head()
    if variant == "E":
        state, mass = chain_variant(src, 1, dt, v, w, head, coords=False, gate=False)
    else:
        coords, gate = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}[variant]
        state, mass = chain_variant(src, steps, dt, v, w, head, coords, gate)
    covs_exp, lanes_exp = expected_scene(lane_pts, dt, v, w)
    rows, lane_err = measure(state, mass, covs_exp, lanes_exp)
    return rows, lane_err


CASES = [  # name, dt, v, omega, steps
    ("straight 13 m/s, 0.5 s in 6 frames", 0.5, 13.0, 0.0, 6),
    ("turning 10 m/s 0.4 rad/s, 0.5 s in 6 frames", 0.5, 10.0, 0.4, 6),
    ("turning, 1.0 s in 12 frames", 1.0, 10.0, 0.4, 12),
    ("radar -> vision hop, 0.04 s in 1 step", 0.04, 25.0, 0.1, 1),
]
NAMES = {"A": "A chained maps", "B": "B maps + gate", "C": "C coordinates", "D": "D coordinates + gate",
         "E": "E one splat (reference)"}

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    print("case | variant | centre error: car ahead / oncoming / crossing / pedestrian (m) | mean IoU cars | lane lateral (m)")
    for name, dt, v, w, steps in CASES:
        for variant in "ABCDE":
            if steps == 1 and variant != "E":
                continue
            rows, lane_err = run(dt, v, w, steps, variant)
            errs = " / ".join(f"{r[1]:.2f}" for r in rows)
            iou = sum(r[2] for r in rows[:3]) / 3
            print(f"{name:45s} | {NAMES[variant]:24s} | {errs:24s} | {iou:.2f} | {lane_err:.3f}")
