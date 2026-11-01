"""scratch: does tests/test_orientation.py catch deliberately broken drawers? (run from the bev_fusion folder)"""
import sys
import numpy as np
import torch
sys.path.insert(0, "tests")
sys.path.insert(0, ".")
sys.path.insert(0, "../autoresearch")
import test_orientation as T
from autogt import view, video
from evaluation import bev_views as V
import flow_viz
from evalsuite import Panel

orig = dict(arrows=view.motion_arrows, wheel=view.colour_wheel, px=video._px, bev=V.bev, frgb=flow_viz.flow_rgb, show=view.show, bevimg=video._bev_image)


def arrows_old(ax, fl):                                           # the sign bug of 2026-10-31
    nx, ny = fl.shape[1:]
    gx = 0.25 + 0.5 * np.arange(nx)[:, None] + 0 * np.arange(ny)[None, :]
    gy = -26 + 0.25 + 0.5 * np.arange(ny)[None, :] + 0 * np.arange(nx)[:, None]
    sel = (np.arange(nx)[:, None] % 6 == 3) & (np.arange(ny)[None, :] % 6 == 3) & (np.hypot(*fl) > 0.3)
    return ax.quiver(gy[sel], gx[sel], -fl[1][sel], fl[0][sel], angles="xy", scale_units="xy", scale=6)


def show_noflip(ax, rgb, title):                                  # forgot the lateral flip of the raster
    ax.imshow(np.flip(rgb, 0), extent=view.EXT, interpolation="nearest")
    ax.set_xlim(26, -26)
    ax.set_ylim(0, 60)


def bev_noinvert(grid, kind, title, data, shapes=(), fov=True, **kw):   # panel without the inverted horizontal axis
    return Panel(kind, title, data, extent=V.extent(grid), xlim=(grid.y_min, grid.y_max), ylim=(grid.x_min, grid.x_max), **kw)


mutants = [
    ("motion arrows with -vy (the old bug)", lambda: setattr(view, "motion_arrows", arrows_old), T.test_autogt_view),
    ("autogt raster without the lateral flip", lambda: setattr(view, "show", show_noflip), T.test_autogt_view),
    ("colour key mirrored left-right", lambda: setattr(view, "colour_wheel", lambda n=41: np.flip(orig["wheel"](n), 1)), T.test_autogt_view),
    ("video overlay mapping with +y to the right", lambda: setattr(video, "_px", lambda x, y: (int(round((26.0 + y) * 9.0)), int(round((60.0 - x) * 9.0)))), T.test_autogt_video),
    ("video bird's-eye image not flipped laterally", lambda: setattr(video, "_bev_image", lambda rgb: orig["bevimg"](np.flip(rgb, 1))), T.test_autogt_video),
    ("evaluation panel without the inverted axis", lambda: setattr(V, "bev", bev_noinvert), T.test_evaluation_panels),
    ("flow_viz picture not flipped laterally", lambda: setattr(flow_viz, "flow_rgb", lambda v, occ=None, vmax=10.0: orig["frgb"](v, occ, vmax)[:, ::-1]), T.test_flow_viz),
]
bad = 0
for name, patch, test in mutants:
    patch()
    try:
        test()
        print("NOT CAUGHT:", name)
        bad += 1
    except AssertionError as e:
        print("caught    :", name, "|", str(e)[:90])
    finally:
        view.motion_arrows, view.colour_wheel, video._px, V.bev = orig["arrows"], orig["wheel"], orig["px"], orig["bev"]
        flow_viz.flow_rgb, view.show, video._bev_image = orig["frgb"], orig["show"], orig["bevimg"]
T.test_autogt_view(); T.test_autogt_video(); T.test_evaluation_panels(); T.test_flow_viz()
print("all mutants caught" if not bad else f"{bad} mutants not caught", "| the unpatched drawers pass again")
