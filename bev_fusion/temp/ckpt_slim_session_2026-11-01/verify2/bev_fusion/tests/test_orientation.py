"""Orientation known-answer test of every bird's-eye drawer of bev_fusion (no data, no GPU, a few seconds).

The convention everywhere: ego frame, x forward (up on the picture), y LEFT (left on the picture), arrays [ix, iy] with iy = 0 at the
RIGHT edge (y = -26 m). A flipped axis, a wrong sign of a lateral component or a colour key that does not match the picture does not
crash anything: the picture is just wrong, as the motion arrows of autogt/view.py were until 2026-10-31. This test puts two markers
into a map, A forward-left (x 45 m, y +15 m, moving 6 m/s forward and 3.6 m/s left) and B backward-right (x 10 m, y -18 m), lets each
drawer draw them and measures where they land on the screen:

  autogt/view.py       raster (show), motion arrows, colour key (colour_wheel) against flow_rgb
  autogt/video.py      the video's bird's-eye image against the pixel mapping of its overlays (_px)
  evaluation/bev_views.py   the evalsuite BEV panel and the box heading line
  tests/flow_viz.py    the flow pictures of the temporal tests and their key

    python tests/test_orientation.py
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in (ROOT, os.path.join(ROOT, ".."), os.path.join(ROOT, "..", "autoresearch"), os.path.join(ROOT, "tests")):   # ".." = the repository root: import autogt
    sys.path.insert(0, p)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

NX, NY, RES = 120, 104, 0.5
A, B = (45.25, 15.25), (10.25, -17.75)                          # (x, y) in metres
VA, VB = (6.0, 3.6), (-3.0, -5.4)                               # (vx, vy) in m/s: forward-left, backward-right
RED, GREEN = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)


def cell(x, y):
    return int(x / RES), int((y + 26.0) / RES)


def marker_map():
    rgb = np.zeros((NX, NY, 3), np.float32)
    for (x, y), col in ((A, RED), (B, GREEN)):
        ix, iy = cell(x, y)
        rgb[ix - 1:ix + 2, iy - 1:iy + 2] = col                  # 3 x 3 cells
    return rgb


def pixels(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int)


def centroid(img, colour, tol=40):
    """(column, row) of the pixels of an exact marker colour in an RGB image, row 0 = top"""
    rows, cols = np.nonzero(np.abs(img - np.array(colour) * 255).sum(-1) < tol)
    assert len(rows) > 20, f"marker {colour} not drawn"
    return cols.mean() + 0.5, rows.mean() + 0.5


def on_screen(fig, ax, y, x):
    """pixel (column, row from the top) of the plot coordinate (horizontal = y, vertical = x)"""
    px, py = ax.transData.transform((y, x))
    return px, fig.canvas.get_width_height()[1] - py


def check_markers(fig, ax, name):
    """both markers sit where their plot coordinates are, and A is up-left, B is down-right of the picture's centre"""
    img = pixels(fig)
    box = ax.get_window_extent()
    cx, cy = (box.x0 + box.x1) / 2, fig.canvas.get_width_height()[1] - (box.y0 + box.y1) / 2
    for (x, y), col, up_left in ((A, RED, True), (B, GREEN, False)):
        got, want = centroid(img, col), on_screen(fig, ax, y, x)
        assert abs(got[0] - want[0]) < 4 and abs(got[1] - want[1]) < 4, f"{name}: marker at x={x} y={y} drawn at {got}, plot coordinates say {want}"
        assert (got[0] < cx and got[1] < cy) == up_left and (got[0] > cx and got[1] > cy) == (not up_left), \
            f"{name}: forward-left must be up-left, backward-right down-right (marker at {got}, centre {(cx, cy)})"


def screen_step(fig, ax, y, x, dy, dx):
    """screen direction (right, up) of a displacement (dy, dx) in plot coordinates"""
    (x0, y0), (x1, y1) = on_screen(fig, ax, y, x), on_screen(fig, ax, y + dy, x + dx)
    return x1 - x0, y0 - y1


def test_autogt_view():
    from autogt import view
    fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
    view.show(ax, marker_map(), "t")
    check_markers(fig, ax, "autogt.view.show")
    plt.close(fig)
    # the motion arrows: a car that drives forward-left points up-left, one that drives backward-right down-right
    fl = np.zeros((2, NX, NY), np.float32)
    for (x, y), (vx, vy) in ((A, VA), (B, VB)):
        ix, iy = cell(x, y)
        fl[0, ix - 1:ix + 2, iy - 1:iy + 2], fl[1, ix - 1:ix + 2, iy - 1:iy + 2] = vx, vy
    fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
    view.show(ax, np.zeros((NX, NY, 3), np.float32), "t")
    q = view.motion_arrows(ax, fl)
    fig.canvas.draw()
    for (px, py), u, v in zip(np.c_[q.X, q.Y], q.U, q.V):
        right, up = screen_step(fig, ax, px, py, u, v)
        want = VA if py > 30 else VB
        assert (right < 0) == (want[1] > 0) and (up > 0) == (want[0] > 0), f"motion arrow at x={py} y={px}: screen step {(right, up)} for v={want}"
    plt.close(fig)
    # the colour key: the picture's colour of a velocity is the key's colour at that velocity (forward up, left left)
    W = view.colour_wheel()
    c = (len(W) - 1) // 2
    for vx, vy in (VA, (0.0, 6.0), (6.0, 0.0), (-6.0, 0.0)):
        want = view.flow_rgb(np.array([[[vx]], [[vy]]], np.float32))[0, 0]
        got = W[int(round(c - vx / view.SAT * c)), int(round(c - vy / view.SAT * c))]
        assert np.abs(got - want).max() < 1e-5, f"colour key: v=({vx}, {vy}) has colour {want}, the key shows {got} at its place"
    assert np.abs(view.flow_rgb(np.array([[[6.0]], [[0.0]]], np.float32)) - view.flow_rgb(np.array([[[0.0]], [[6.0]]], np.float32))).max() > 0.3


def test_autogt_video():
    from autogt import video
    img = video._bev_image(marker_map())                                         # BGR
    for (x, y), col in ((A, (0.0, 0.0, 1.0)), (B, GREEN)):                       # red and green in the BGR order of the image
        got, want = centroid(img.astype(int), col), video._px(x, y)
        assert abs(got[0] - want[0]) < 6 and abs(got[1] - want[1]) < 6, f"video: marker x={x} y={y} drawn at {got}, the overlays use {want}"
    assert video._px(*A)[0] < video._px(*B)[0] and video._px(*A)[1] < video._px(*B)[1]


def test_evaluation_panels():
    from evalsuite.render import figure
    from autogt.grid import GridSpec
    from evaluation import bev_views as V
    grid = GridSpec()
    fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
    figure._panel(fig, ax, V.bev(grid, "rgb", "t", marker_map(), fov=False))
    check_markers(fig, ax, "evaluation.bev_views.bev panel")
    plt.close(fig)
    # a box at yaw +0.6 rad (turned to the left): its heading line leaves the centre forward and to the left, along the long edge
    shapes = V.box_shapes([dict(cx=30.0, cy=-8.0, length=4.0, width=2.0, yaw=0.6)], "r")
    poly, line = shapes[0].xy, np.asarray(shapes[1].xy, float)                   # plot coordinates (y, x)
    head = line[1] - line[0]
    assert head[0] > 0 and head[1] > 0 and abs(math.atan2(head[0], head[1]) - 0.6) < 1e-6, f"heading line {line}"
    assert np.allclose(np.mean(poly, 0), (-8.0, 30.0), atol=1e-6)
    p = np.asarray(poly, float)                                                  # corners: front-left, front-right, rear-right, rear-left
    side, front = p[0] - p[3], p[0] - p[1]                                       # the car's left side (its length) and its front (its width)
    assert abs(side[0] * head[1] - side[1] * head[0]) < 1e-6 and side @ head > 0, "the long side runs along the heading, front first"
    assert abs(np.hypot(*side) - 4.0) < 1e-6 and abs(np.hypot(*front) - 2.0) < 1e-6
    assert front[0] * head[1] - front[1] * head[0] != 0 and abs(front @ head) < 1e-6, "the front edge is perpendicular to the heading"


def test_flow_viz():
    import flow_viz
    v = torch.zeros(2, NX, NY)
    for (x, y), (vx, vy) in ((A, VA), (B, VB)):
        ix, iy = cell(x, y)
        v[0, ix - 1:ix + 2, iy - 1:iy + 2], v[1, ix - 1:ix + 2, iy - 1:iy + 2] = vx, vy
    img = flow_viz.flow_rgb(v, vmax=12.0)                                         # for imshow(origin="lower"): row = x, column 0 = left
    rows, cols = np.nonzero(img.sum(-1) > 0.05)
    up_left = (rows * RES > 30) & (cols < NY / 2)
    assert up_left.any() and ((rows * RES < 30) & (cols > NY / 2)).any() and (up_left | ((rows * RES < 30) & (cols > NY / 2))).all(), "flow_viz.flow_rgb"
    ix, iy = cell(*A)
    assert img[ix, NY - 1 - iy].sum() > 0.1 and img[ix, iy].sum() == 0
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    flow_viz.wheel(ax, vmax=12.0)
    shot = pixels(fig)
    for vx, vy in (VA, VB, (0.0, 8.0), (8.0, 0.0)):
        want = flow_viz.flow_rgb(torch.tensor([[[vx]], [[vy]]], dtype=torch.float32), vmax=12.0)[0, 0] * 255
        px, py = on_screen(fig, ax, vy, vx)
        got = shot[int(round(py)), int(round(px))]
        assert np.abs(got - want).sum() < 60, f"flow_viz key: v=({vx}, {vy}) has colour {want}, the key shows {got} at its place"
    plt.close(fig)


if __name__ == "__main__":
    for t in (test_autogt_view, test_autogt_video, test_evaluation_panels, test_flow_viz):
        t()
        print("ok", t.__name__)
    print("orientation test ok: raster, arrows, colour keys and heading lines of autogt, the video, the evaluation panels and flow_viz "
          "agree (x up, +y left, forward-left = up-left)")
