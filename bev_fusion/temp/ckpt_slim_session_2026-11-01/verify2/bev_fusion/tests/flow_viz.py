"""Flow-field pictures for the temporal tests: velocity as colour (hue = direction, brightness = speed) and the colour
wheel that explains it. Shared by tests/test_flow_refine.py and tests/test_flow_explains.py.
BEV convention of the pictures: x forward = up, +y (left) = left, i.e. imshow(..., origin="lower") of a map flipped
along y.
"""
import math

import torch


def flow_rgb(v, occ=None, vmax=10.0):
    """v (2, X, Y) m/s -> (X, Y, 3) image array (flipped for imshow origin="lower"); cells with occ = 0 black."""
    from matplotlib.colors import hsv_to_rgb
    ang = (torch.atan2(v[1], v[0]) / (2 * math.pi)) % 1.0
    val = (v.norm(dim=0) / vmax).clamp(0, 1)
    rgb = torch.tensor(hsv_to_rgb(torch.stack([ang, torch.ones_like(ang), val], -1).numpy()))
    if occ is not None:
        rgb = rgb * occ.unsqueeze(-1)
    return rgb.flip(1).numpy()


def wheel(ax, vmax=10.0, n=101):
    """draw the colour key into a matplotlib axis: direction (forward = up) and speed (0 at the centre, vmax at the
    rim)."""
    g = torch.linspace(-vmax, vmax, n)
    vx, vy = torch.meshgrid(g, g, indexing="ij")                      # vx up (rows), vy left (columns, flipped)
    inside = (vx ** 2 + vy ** 2 <= vmax ** 2).float()
    ax.imshow(flow_rgb(torch.stack([vx, vy]), inside, vmax), origin="lower", extent=(vmax, -vmax, -vmax, vmax))
    ax.set_title(f"key: forward = up, left = left; rim {vmax:.0f} m/s", fontsize=7)
    ax.axis("off")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # self-test: a forward velocity is red-ish at the top of the wheel, the centre is black
    img = flow_rgb(torch.tensor([[[10.0]], [[0.0]]]))
    assert img[0, 0, 0] > 0.9 and img[0, 0, 1] < 0.1 and img[0, 0, 2] < 0.1, img
    assert flow_rgb(torch.zeros(2, 1, 1)).max() == 0
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    wheel(ax)
    fig.savefig("temp/checks/flow_wheel.png", dpi=100)
    print("flow_viz selftest ok (forward = red, zero = black); wheel temp/checks/flow_wheel.png")
