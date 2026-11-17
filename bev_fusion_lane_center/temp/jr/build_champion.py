"""Assemble the report of the end-to-end token champion (D113) from the part files; same renderer as build_ergebnis.py.
Run (WSL): BFLC_ROOT=/mnt/c/Auto-research/bev_fusion_lane_center BFLC_OUT=<abs html> /home/ml/miniconda3/envs/gcar/bin/python temp/jr/build_champion.py
Parts: results/Ergebnis_E2E_Champion_2026-11-17/parts/*.md (+ 02_architecture.svg, 04_images/*.jpg). Output: one self-contained html."""
import base64, os, re, sys
from markdown_it import MarkdownIt

ROOT = os.path.join(os.environ.get("BFLC_ROOT", os.getcwd()), "results", "Ergebnis_E2E_Champion_2026-11-17")
P = os.path.join(ROOT, "parts")
OUT = os.environ.get("BFLC_OUT") or os.path.join(ROOT, "E2E_Champion_Bericht.html")
md = MarkdownIt("commonmark", {"html": False}).enable("table")


def read(name, default=""):
    p = os.path.join(P, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else default


def render(text, demote=1):
    if demote:                                           # the section title comes from the builder: drop the part's own first heading
        text = re.sub(r"\A\s*#+ [^\n]*\n", "", text)
    text = re.sub(r"^(#+) ", lambda m: "#" * min(6, len(m.group(1)) + demote) + " ", text, flags=re.M)
    html = md.render(text)

    def img(m):
        src = m.group(1)
        f = os.path.join(P, src)
        if os.path.exists(f):
            raw = open(f, "rb").read()
            try:                                           # mail size: JPEG quality 68, optimised, progressive
                import io
                from PIL import Image
                buf = io.BytesIO()
                Image.open(io.BytesIO(raw)).convert("RGB").save(buf, "JPEG", quality=68, optimize=True, progressive=True, subsampling=2)
                raw = buf.getvalue()
            except Exception:
                pass
            return 'src="data:image/jpeg;base64,%s"' % base64.b64encode(raw).decode()
        return m.group(0)
    html = re.sub(r'src="([^"]+\.(?:jpg|jpeg))"', img, html)
    html = html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    html = html.replace("PENDING", '<span class="pend">PENDING</span>')
    return html


svg = read("02_architecture.svg")
svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg)
sections = [
    ("start", "Start here", render(read("00a_start.md"))),
    ("summary", "Summary", render(read("00_summary.md"), 0)),
    ("gt", "1. Ground truth: what is needed, why, how it is made, how to rebuild it", render(read("01_gt.md"))),
    ("arch", "2. System architecture", '<div class="fig">%s</div>' % svg + render(read("02_architecture.md"))),
    ("results", "3. Experiments, results, compute, difficult situations", render(read("03_results.md"))),
    ("qual", "4. Qualitative evaluation", render(read("04_qualitative.md"))),
    ("code", "5. Champion code", render(read("07_code.md"))),
    ("open", "6. What is still hard, open decisions and next steps", render(read("08_open.md"))),
    ("rebuild", "7. Rebuilding it: from nuScenes to the numbers", render(read("09_rebuild.md"))),
]
nav = " | ".join('<a href="#%s">%s</a>' % (k, t.split(".")[0] if t[0].isdigit() else t) for k, t, _ in sections)
body = "\n".join('<section id="%s"><h2>%s</h2>%s</section>' % (k, t, h) for k, t, h in sections if h.strip())
css = """
:root{--bg:#fbfaf7;--fg:#1d2330;--mut:#5b6475;--ln:#d9d6cc;--ac:#1f5fa8;--bgc:#f1efe8;--pe:#fff1c9;--pf:#7a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#161a22;--fg:#e6e8ee;--mut:#9aa3b5;--ln:#2d3443;--ac:#6aa7ee;--bgc:#1e2330;--pe:#4a3d12;--pf:#ffe08a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif;margin:0}
main{max-width:980px;margin:0 auto;padding:18px 16px 60px}
h1{font-size:24px;margin:.2em 0}h2{font-size:19px;margin:1.6em 0 .5em;border-bottom:1px solid var(--ln);padding-bottom:4px}
h3{font-size:16px;margin:1.2em 0 .3em}h4{font-size:14px;margin:1em 0 .2em}
code{background:var(--bgc);padding:1px 4px;border-radius:3px;font-size:.88em}
nav{font-size:13px;color:var(--mut);margin:6px 0 14px}nav a{color:var(--ac);text-decoration:none}
.tw{overflow-x:auto;margin:.6em 0}table{border-collapse:collapse;font-size:12.5px}th,td{border:1px solid var(--ln);padding:3px 6px;vertical-align:top}th{background:var(--bgc);text-align:left}
img{max-width:100%;height:auto;border:1px solid var(--ln)}.fig svg{max-width:100%;height:auto;background:#fbfaf7}
p,li{max-width:80ch}.sub{color:var(--mut);font-size:13px}
.pend{background:var(--pe);color:var(--pf);font-weight:700;padding:0 4px;border-radius:3px;font-size:.85em}
"""
html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>E2E token champion: report</title><style>%s</style></head><body><main>'
        '<h1>An end-to-end differentiable lane-centre graph from camera + radar: the token champion</h1>'
        '<div class="sub">Spurmitte project. SKELETON of 2026-11-17: every cell or paragraph marked PENDING is filled when the champion exists. Numbers come from stored cards and ledger rows; VAL = 873 frames / 22 scenes, TEST untouched.</div>'
        '<nav>%s</nav>%s</main></body></html>') % (css, nav, body)
open(OUT, "w", encoding="utf-8").write(html)
print(OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
