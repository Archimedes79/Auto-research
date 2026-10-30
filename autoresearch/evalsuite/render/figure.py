"""The figure engine: a Page (title, panels, overlays, legend) -> an image. One style for every view of every network.

Panel kinds: image | raster | rgb | text | table | bars (see contracts.Panel). The error colour code is the same everywhere:
a cell both predicted and in the ground truth is grey, predicted but not in the ground truth is RED (false positive), in the
ground truth but not predicted is BLUE (false negative), not evaluated is cream, correct background is white
(`error_rgb`, `ERROR_LEGEND`).
"""
from __future__ import annotations

import textwrap
from typing import Optional, Tuple

import numpy as np

from ..contracts import Page, Panel, Shape

TP_COLOR = (0.70, 0.70, 0.70)
FP_COLOR = (0.90, 0.20, 0.20)
FN_COLOR = (0.20, 0.40, 0.95)
SKIP_COLOR = (0.95, 0.95, 0.88)
TN_COLOR = (1.0, 1.0, 1.0)
ERROR_LEGEND = [("true positive", TP_COLOR), ("false positive", FP_COLOR), ("false negative", FN_COLOR),
                ("not evaluated", SKIP_COLOR)]


def error_rgb(pred, gt, valid=None) -> np.ndarray:
    """(H, W, 3) float colour image of a binary prediction against a binary ground truth on the evaluated cells (`valid`,
    default all): TP grey, FP red, FN blue, TN white, not evaluated cream."""
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    v = np.ones_like(p) if valid is None else np.asarray(valid).astype(bool)
    img = np.empty(p.shape + (3,), np.float32)
    img[:] = TN_COLOR
    img[p & g & v] = TP_COLOR
    img[p & ~g & v] = FP_COLOR
    img[~p & g & v] = FN_COLOR
    img[~v] = SKIP_COLOR
    return img


def error_counts(pred, gt, valid=None) -> Tuple[int, int, int]:
    """(tp, fp, fn) cell counts of the same comparison (for captions)"""
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    v = np.ones_like(p) if valid is None else np.asarray(valid).astype(bool)
    return int((p & g & v).sum()), int((p & ~g & v).sum()), int((~p & g & v).sum())


def _table_text(rows) -> str:
    rows = [[("" if c is None else (f"{c:.3g}" if isinstance(c, float) else str(c))) for c in r] for r in rows]
    if not rows:
        return ""
    w = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(max(len(r) for r in rows))]
    lines = ["  ".join((r[i] if i < len(r) else "").ljust(w[i]) for i in range(len(w))).rstrip() for r in rows]
    if len(lines) > 1:
        lines.insert(1, "-" * min(len(lines[0]), 90))
    return "\n".join(lines)


def _shape(ax, s: Shape) -> None:
    from matplotlib.patches import Circle, Polygon
    xy = np.asarray(s.xy, float)
    if s.kind == "polygon":
        ax.add_patch(Polygon(xy, closed=True, fill=False, ec=s.color, ls=s.ls, lw=s.lw))
    elif s.kind == "line":
        ax.plot(xy[:, 0], xy[:, 1], color=s.color, ls=s.ls, lw=s.lw)
    elif s.kind == "points":
        ax.scatter(xy[:, 0], xy[:, 1], s=max(2.0, 6 * s.lw), c=[s.color] if not isinstance(s.color, (list, np.ndarray)) else s.color)
    elif s.kind == "circle":
        ax.add_patch(Circle((xy[0], xy[1]), xy[2], fill=False, ec=s.color, ls=s.ls, lw=s.lw))
    elif s.kind == "arrow":
        ax.annotate("", xy=(xy[2], xy[3]), xytext=(xy[0], xy[1]),
                    arrowprops=dict(arrowstyle="->", color=s.color, lw=s.lw, ls=s.ls))
    elif s.kind == "text":
        ax.text(xy[0], xy[1], s.label, color=s.color, fontsize=6, ha="center", va="center")
    else:
        raise ValueError(f"unknown shape kind {s.kind!r}")


def _panel(fig, ax, p: Panel) -> None:
    title = textwrap.fill(p.title, 48)
    if p.kind == "image":
        a = np.asarray(p.data)
        ax.imshow(a, cmap="gray" if a.ndim == 2 else None)
        ax.axis("off")
    elif p.kind in ("raster", "rgb"):
        a = np.asarray(p.data)
        kw = dict(origin=p.origin, interpolation="nearest", aspect="equal" if p.equal else "auto")
        if p.extent is not None:
            kw["extent"] = p.extent
        if p.kind == "raster":
            finite = a[np.isfinite(a)] if a.size else a
            vmin = p.vmin if p.vmin is not None else (float(finite.min()) if finite.size else 0.0)
            vmax = p.vmax if p.vmax is not None else (float(np.percentile(finite, 99.5)) if finite.size else 1.0)
            im = ax.imshow(a, cmap=p.cmap, vmin=vmin, vmax=max(vmax, vmin + 1e-9), **kw)
            if p.colorbar:
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01).ax.tick_params(labelsize=6)
        else:
            ax.imshow(np.clip(a, 0, 1), **kw)
        if p.xlim is not None:
            ax.set_xlim(*p.xlim)
        if p.ylim is not None:
            ax.set_ylim(*p.ylim)
        ax.tick_params(labelsize=6)
    elif p.kind in ("text", "table"):
        ax.axis("off")
        text = p.data if p.kind == "text" else _table_text(p.data)
        ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=6.5, transform=ax.transAxes)
    elif p.kind == "bars":
        items = list(dict(p.data).items())
        ax.barh([k for k, _ in items][::-1], [float(v) for _, v in items][::-1], color="tab:blue")
        ax.tick_params(labelsize=7)
    else:
        raise ValueError(f"unknown panel kind {p.kind!r}")
    view = (ax.get_xlim(), ax.get_ylim()) if p.kind in ("image", "raster", "rgb") else None
    for s in p.shapes:
        _shape(ax, s)
    if view is not None:                               # the overlays (which may reach far outside) must not move the limits
        ax.set_xlim(*(p.xlim if p.xlim is not None else view[0]))
        ax.set_ylim(*(p.ylim if p.ylim is not None else view[1]))
    ax.set_title(title, fontsize=8)
    if p.note and p.kind in ("raster", "rgb", "image"):          # what to look for, under the picture
        ax.set_xlabel(textwrap.fill(p.note, 60), fontsize=6, color="0.35")


def render_page(page: Page, path: Optional[str] = None, dpi: int = 90) -> np.ndarray:
    """Draw a Page; returns the image (H, W, 3 uint8) and writes it to `path` (png) when given."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    n = max(1, len(page.panels))
    ncols = max(1, min(page.ncols, n))
    nrows = -(-n // ncols)
    pw, ph = page.panel_size
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * pw, nrows * ph + 0.7), squeeze=False, dpi=dpi)
    for i, ax in enumerate(axs.ravel()):
        if i < len(page.panels):
            _panel(fig, ax, page.panels[i])
        else:
            ax.axis("off")
    if page.legend:
        fig.legend(handles=[Patch(color=c, label=l) for l, c in page.legend], loc="lower center", ncol=min(len(page.legend), 6),
                   fontsize=8, frameon=False)
    if page.footer:
        fig.text(0.01, 0.005, page.footer, fontsize=7, va="bottom", ha="left", family="monospace")
    fig.suptitle("\n".join(textwrap.fill(line, 150) for line in page.title.split("\n")), fontsize=10)
    fig.tight_layout(rect=(0, 0.05 if (page.legend or page.footer) else 0, 1, 0.96))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return img
