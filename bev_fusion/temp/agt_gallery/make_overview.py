"""scratch: build the GT quality overview PDF from the rendered pictures.

  make_overview.py OUT.pdf   (reads temp/agt_gallery/{overview.json, picks.json, pics/, close/, thumbs_*.png})

overview.json: {"pages": [{"title", "lines", "size", "wrap"}], "notes": {"<scene>_k<kk>": caption}, "extra": [{"png", "title", "note"}], "tail": [pages]}
Order: text pages, thumbnail sheets, the examples (three per page), the close-ups (one per page), the closing pages.
"""
import glob
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

W = 16.5
ov = json.load(open("temp/agt_gallery/overview.json", encoding="utf-8"))
picks = json.load(open("temp/agt_gallery/picks.json"))
notes = ov.get("notes", {})


def text_page(pdf, title, lines, size=9, wrap=175, per=68):
    wrapped = []
    for ln in lines:
        wrapped.extend(textwrap.wrap(ln, wrap, subsequent_indent="    ", drop_whitespace=False) or [""])
    for ci in range(0, max(len(wrapped), 1), per):
        fig = plt.figure(figsize=(W, 11.7))
        fig.text(0.03, 0.97, title + (f" (page {ci // per + 1})" if len(wrapped) > per else ""), fontsize=15, va="top", weight="bold")
        fig.text(0.03, 0.925, "\n".join(wrapped[ci:ci + per]), fontsize=size, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close(fig)


def image_page(pdf, items, title=None):
    """items: [(png path, headline, note)] stacked on one page whose height follows the pictures."""
    ims = [Image.open(p).convert("RGB") for p, _, _ in items]
    hs = [W * 0.99 * im.height / im.width for im in ims]
    top = 0.55 if title else 0.15
    H = top + sum(h + 0.85 for h in hs)
    fig = plt.figure(figsize=(W, H))
    if title:
        fig.text(0.02, 1 - 0.12 / H, title, fontsize=13, va="top", weight="bold")
    y = H - top
    for (png, head, note), im, h_in in zip(items, ims, hs):
        ax = fig.add_axes([0.005, (y - 0.62 - h_in) / H, 0.99, h_in / H])
        ax.imshow(im)
        ax.axis("off")
        fig.text(0.01, (y - 0.04) / H, head, fontsize=11, va="top", weight="bold")
        fig.text(0.01, (y - 0.30) / H, "\n".join(textwrap.wrap(note, 215)[:2]), fontsize=8.5, va="top", color="#333333")
        y -= h_in + 0.85
    pdf.savefig(fig)
    plt.close(fig)


out = sys.argv[1]
with PdfPages(out) as pdf:
    for pg in ov["pages"]:
        text_page(pdf, pg["title"], pg["lines"], pg.get("size", 9), pg.get("wrap", 175))
    for i, png in enumerate(sorted(glob.glob("temp/agt_gallery/thumbs_*.png"))):
        im = Image.open(png).convert("RGB")
        fig = plt.figure(figsize=(W, W * im.height / im.width + 0.6))
        H = fig.get_figheight()
        ax = fig.add_axes([0.0, 0.0, 1.0, (H - 0.6) / H])
        ax.imshow(im)
        ax.axis("off")
        fig.text(0.01, 1 - 0.1 / H, f"random keyframes (seeded draw, not selected), sheet {i + 1}: semantic BEV with objects (red vehicles, magenta "
                 "pedestrians, orange two-wheelers, yellow = ignored, cyan arrows = velocity > 1 m/s, dark = not seen)", fontsize=10, va="top")
        pdf.savefig(fig)
        plt.close(fig)
    items = []
    for p in picks["examples"]:
        key = f"{p['scene']}_k{p['k']:02d}"
        png = f"temp/agt_gallery/pics/{key}.png"
        if os.path.exists(png):
            items.append((png, f"{p['title']}: {p['scene']} keyframe {p['k']} ({p['loc']}) - {p['desc'][:90]}", notes.get(key, p["note"])))
    for i in range(0, len(items), 3):
        image_page(pdf, items[i:i + 3], "examples (4 panels: camera with boxes + map lines | semantic BEV | objects + velocity | motion relative to the static world)" if i == 0 else None)
    for e in ov.get("extra", []):
        image_page(pdf, [(e["png"], e["title"], e["note"])], "full-resolution close-ups: GT map lines and boxes on the camera image" if e is ov["extra"][0] else None)
    for pg in ov.get("tail", []):
        text_page(pdf, pg["title"], pg["lines"], pg.get("size", 9), pg.get("wrap", 175))
print("written", out, round(os.path.getsize(out) / 1e6, 1), "MB")
