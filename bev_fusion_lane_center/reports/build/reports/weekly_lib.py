"""Shared helpers for the two weekly reports (HTML, A4) - run in the gcar env (Pillow, matplotlib).

    import weekly_lib as W
    d = W.Doc("Title", "subtitle", "Week 1")      # collects HTML parts
    d.h1("..."); d.p("..."); d.fig("figures/x.png", "caption"); d.table(rows, header=[...])
    d.save("/mnt/c/.../out.html")                  # then print with reports/build/print_weekly.ps1

Images are resized once into a temp cache (/tmp/weekly_img_cache) (max 1000 px, JPEG for photos, PNG for line art) and embedded
as base64, so the HTML is a single file.
"""
import base64
import hashlib
import html
import io
import os

from PIL import Image

ROOT = "/mnt/c/ML-Experiment/bev_fusion_lane_center"
CACHE = "/tmp/weekly_img_cache"  # regenerated on each build
esc = html.escape

CSS = """
@page { size: A4; margin: 15mm 14mm 17mm 14mm; @bottom-center { content: counter(page); font: 8pt "Segoe UI", Arial, sans-serif; color: #5b6570; } }
@page :first { @bottom-center { content: ""; } }
:root { --ink:#1b1f24; --mute:#5b6570; --navy:#14213D; --blue:#1d4ea3; --orange:#c85f00; --green:#1a7f3c; --amber:#9a6a00; --red:#b3261e; --line:#d5dae0; --bg:#f4f6f9; --tint:#e8eefb; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; color: var(--ink); font-size: 9.6pt; line-height: 1.42; margin: 0; }
.cover { background: var(--navy); color: #fff; padding: 16mm 14mm 12mm 14mm; margin: -15mm -14mm 6mm -14mm; page-break-after: avoid; }
.cover .kicker { font-size: 9pt; letter-spacing: 1.2pt; text-transform: uppercase; color: #aebfe6; margin-bottom: 4mm; }
.cover h1 { font-family: Cambria, Georgia, serif; font-size: 27pt; line-height: 1.12; margin: 0 0 3mm 0; color: #fff; border: 0; padding: 0; }
.cover .sub { font-size: 11pt; color: #d6e0f7; max-width: 150mm; }
.cover .meta { margin-top: 6mm; font-size: 8.5pt; color: #aebfe6; }
h1 { font-family: Cambria, Georgia, serif; font-size: 17pt; color: var(--navy); margin: 0 0 2.5mm 0; line-height: 1.18; page-break-after: avoid; }
h1.chap { margin-top: 0; padding-top: 0; }
h2 { font-size: 11.6pt; color: var(--blue); margin: 5mm 0 1.5mm 0; page-break-after: avoid; }
h3 { font-size: 9.8pt; color: var(--navy); margin: 3mm 0 0.8mm 0; page-break-after: avoid; }
p { margin: 0 0 2mm 0; }
ul, ol { margin: 0 0 2mm 0; padding-left: 5mm; } li { margin: 0 0 0.9mm 0; }
a { color: var(--blue); text-decoration: none; }
table.keep { page-break-inside: avoid; }
.callout, .stat, .stats { page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.3; margin: 1.5mm 0 3mm 0; }
th { text-align: left; background: var(--bg); border-bottom: 1.4px solid var(--line); padding: 1.1mm 1.6mm; font-weight: 600; color: var(--navy); }
td { border-bottom: 1px solid var(--line); padding: 0.9mm 1.6mm; vertical-align: top; }
tr { page-break-inside: avoid; }
td.n, th.n { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
td.id { white-space: nowrap; font-weight: 600; color: var(--navy); }
tr.hl td { background: var(--tint); font-weight: 600; }
tr.neg td { color: var(--mute); }
figure { margin: 2mm 0 3mm 0; page-break-inside: avoid; } figure img { width: 100%; display: block; border: 1px solid var(--line); }
figcaption { font-size: 8pt; color: var(--mute); margin-top: 1mm; }
.stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; margin: 2mm 0 4mm 0; }
.stat { background: var(--bg); border-radius: 2mm; padding: 3mm 3.5mm; }
.stat .v { font-family: Cambria, Georgia, serif; font-size: 20pt; line-height: 1.05; color: var(--blue); }
.stat .l { font-size: 8.4pt; font-weight: 600; color: var(--navy); margin-top: 1mm; }
.stat .d { font-size: 7.8pt; color: var(--mute); margin-top: 0.6mm; line-height: 1.3; }
.callout { background: var(--tint); border-radius: 2mm; padding: 2.5mm 3.5mm; margin: 2mm 0 3mm 0; }
.callout.neg { background: #fbeceb; } .callout.ok { background: #e6f4ea; } .callout.warn { background: #fdf3dc; }
.callout b.t { display: block; margin-bottom: 0.6mm; color: var(--navy); }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; } .two > div { min-width: 0; }
.tag { display: inline-block; font-size: 7.4pt; font-weight: 700; padding: 0 1.4mm; border-radius: 1mm; border: 1px solid; vertical-align: 1px; }
.tag.ok { color: var(--green); border-color: var(--green); background: #e6f4ea; }
.tag.neg { color: var(--red); border-color: var(--red); background: #fbeceb; }
.tag.tie { color: var(--amber); border-color: var(--amber); background: #fdf3dc; }
.tag.info { color: var(--blue); border-color: var(--blue); background: var(--tint); }
.small { font-size: 8pt; color: var(--mute); }
.pb { page-break-before: always; }
.toc { columns: 2; column-gap: 8mm; font-size: 9pt; } .toc div { margin-bottom: 1mm; } .toc .n { color: var(--mute); }
.timeline td { font-size: 8.2pt; }
code { font-family: Consolas, monospace; font-size: 8.4pt; background: var(--bg); padding: 0 1mm; border-radius: 0.8mm; }
"""


def img_uri(path):
    """Resized, embedded copy of an image (path relative to ROOT or absolute)."""
    p = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.md5((p + str(os.path.getmtime(p))).encode()).hexdigest()[:12]
    im = Image.open(p)
    has_alpha = im.mode in ("RGBA", "LA", "P")
    ext = "png" if has_alpha or p.lower().endswith(".png") and im.size[0] * im.size[1] < 700 * 700 else "jpg"
    out = f"{CACHE}/{os.path.splitext(os.path.basename(p))[0]}_{key}.{ext}"
    if not os.path.exists(out):
        im = im.convert("RGBA") if ext == "png" else im.convert("RGB")
        if im.size[0] > 1000:
            im = im.resize((1000, int(im.size[1] * 1000 / im.size[0])), Image.LANCZOS)
        im.save(out, optimize=True) if ext == "png" else im.save(out, quality=78, optimize=True)
    with open(out, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f"data:image/{'png' if ext == 'png' else 'jpeg'};base64,{b}"


class Doc:
    def __init__(self, title, subtitle, kicker, meta=""):
        self.title, self.subtitle, self.kicker, self.meta = title, subtitle, kicker, meta
        self.parts = []
        self.toc = []

    def raw(self, s):
        self.parts.append(s)

    def h1(self, t, new_page=False, toc=True):
        i = len(self.toc)
        if toc:
            self.toc.append((i, t))
        self.parts.append(f'<h1 class="chap{" pb" if new_page else ""}" id="s{i}">{esc(t)}</h1>')

    def h2(self, t):
        self.parts.append(f"<h2>{esc(t)}</h2>")

    def h3(self, t):
        self.parts.append(f"<h3>{esc(t)}</h3>")

    def p(self, t, cls=""):
        self.parts.append(f'<p class="{cls}">{t}</p>')

    def ul(self, items, ordered=False):
        tg = "ol" if ordered else "ul"
        self.parts.append(f"<{tg}>" + "".join(f"<li>{x}</li>" for x in items) + f"</{tg}>")

    def callout(self, title, body, kind=""):
        self.parts.append(f'<div class="callout {kind}"><b class="t">{title}</b>{body}</div>')

    def stats(self, items):
        """items: (value, label, detail)"""
        self.parts.append('<div class="stats">' + "".join(
            f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div><div class="d">{d}</div></div>' for v, l, d in items) + "</div>")

    def fig(self, path, caption, width=100):
        st = f' style="width:{width}%;margin-left:auto;margin-right:auto"' if width != 100 else ""
        self.parts.append(f'<figure{st}><img src="{img_uri(path)}"><figcaption>{caption}</figcaption></figure>')

    def table(self, rows, header=None, num=(), hl=(), neg=(), widths=None, idcol=0):
        """rows: list of lists (cells are HTML); num: column indices right-aligned; hl/neg: row indices."""
        h = "<table class=\"keep\">" if len(rows) <= 12 else "<table>"
        if header:
            h += "<tr>" + "".join(
                f'<th class="{"n" if j in num else ""}"{f" style=width:{widths[j]}" if widths and widths[j] else ""}>{c}</th>'
                for j, c in enumerate(header)) + "</tr>"
        for i, r in enumerate(rows):
            cls = "hl" if i in hl else ("neg" if i in neg else "")
            h += f'<tr class="{cls}">' + "".join(
                f'<td class="{"n" if j in num else ("id" if j == idcol and idcol is not None else "")}">{c}</td>' for j, c in enumerate(r)) + "</tr>"
        self.parts.append(h + "</table>")

    def two(self, left, right):
        self.parts.append(f'<div class="two"><div>{left}</div><div>{right}</div></div>')

    @staticmethod
    def tag(kind, text):
        return f'<span class="tag {kind}">{text}</span>'

    def save(self, out):
        toc = "".join(f'<div><a href="#s{i}">{esc(t)}</a></div>' for i, t in self.toc)
        cover = (f'<div class="cover"><div class="kicker">{esc(self.kicker)}</div><h1>{esc(self.title)}</h1>'
                 f'<div class="sub">{self.subtitle}</div><div class="meta">{self.meta}</div></div>')
        body = cover + "".join(self.parts)
        body = body.replace("<!--TOC-->", f'<div class="toc">{toc}</div>')
        doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{esc(self.title)}</title>'
               f"<style>{CSS}</style></head><body>{body}</body></html>")
        with open(out, "w", encoding="utf-8") as f:
            f.write(doc)
        print("written", out, f"{len(doc) / 1e6:.1f} MB")
