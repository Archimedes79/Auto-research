"""Result images for lane graphs, built on viz_graph (ground truth green, prediction red, ego frame: x right, y forward, +-50 m).

  render_frame(pred, gt, path, title=..., metrics=..., background=...)   one panel -> png
  render_compare({"rule": g1, "learned": g2}, gt, path, ...)             one panel per method on the same frame -> png
  render_grid(preds, gts, ids, path, ncol=4)                             many frames in one sheet -> png
  frame_array(pred, gt, ...)                                             the same panel as an RGB uint8 array (used by the video tool)
  montage(paths, out, ncol)                                              stitch existing png files into one sheet
Graph format: dict(pts=[(n,2) arrays], edges=[(i,j)]) (see evaluation/sparse_lanes/README.md).
Usage: tools/env/_py14.sh viz/render.py --pred pred.pkl --gt gt.pkl --ids 0,10,20 --out sheet.png
"""
import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import viz_graph as VG  # noqa: E402


def _fig(n, size, ncol=None):
    ncol = ncol or n
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow + 0.35))
    return fig, np.atleast_1d(axs).ravel()


def _title(title, metrics):
    if metrics:
        return (title + "\n" if title else "") + "  ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items())
    return title or ""


def render_frame(pred, gt, path, title="", metrics=None, background=None, size=5.0, dpi=110):
    fig, axs = _fig(1, size)
    VG.panel(axs[0], pred, gt, title=_title(title, metrics), background=background)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def render_compare(preds, gt, path, titles=None, metrics=None, size=4.2, dpi=110):
    names = list(preds)
    fig, axs = _fig(len(names), size)
    for ax, nm in zip(axs, names):
        VG.panel(ax, preds[nm], gt, title=_title((titles or {}).get(nm, nm), (metrics or {}).get(nm)))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def render_grid(preds, gts, ids, path, ncol=4, titles=None, size=3.4):
    VG.grid(preds, gts, list(ids), path, titles=titles, ncol=ncol, figsize=size)
    return path


def frame_array(pred, gt, title="", metrics=None, background=None, size=5.0, dpi=80):
    fig, axs = _fig(1, size)
    VG.panel(axs[0], pred, gt, title=_title(title, metrics), background=background)
    fig.tight_layout()
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return a


def montage(paths, out, ncol=3, pad=6):
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(i.width for i in ims), max(i.height for i in ims)
    nrow = int(np.ceil(len(ims) / ncol))
    sheet = Image.new("RGB", (ncol * w + (ncol + 1) * pad, nrow * h + (nrow + 1) * pad), "white")
    for k, im in enumerate(ims):
        sheet.paste(im, (pad + (k % ncol) * (w + pad), pad + (k // ncol) * (h + pad)))
    sheet.save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pickle: list of graphs")
    ap.add_argument("--gt", required=True, help="pickle: list of graphs")
    ap.add_argument("--ids", default="0,1,2,3", help="frame indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ncol", type=int, default=4)
    a = ap.parse_args()
    P, G = pickle.load(open(a.pred, "rb")), pickle.load(open(a.gt, "rb"))
    ids = [int(x) for x in a.ids.split(",")]
    render_grid([P[i] for i in ids], [G[i] for i in ids], ids, a.out, ncol=a.ncol)
    print("written", os.path.abspath(a.out))
