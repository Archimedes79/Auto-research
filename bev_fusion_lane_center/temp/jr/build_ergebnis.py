"""Assemble the short, mailable results report (2026-11-15) from the part files.
Run: wsl bash -c 'cd /mnt/c/Auto-research/bev_fusion_lane_center && /home/ml/miniconda3/envs/gcar/bin/python temp/jr/build_ergebnis.py'
Parts: results/Ergebnis_Spurmitte_2026-11-15/parts/*.md (+ 02_architecture.svg, 04_images/*.jpg). Output: one self-contained html."""
import base64, os, re, sys
from markdown_it import MarkdownIt

ROOT = os.path.join(os.environ.get("BFLC_ROOT", os.getcwd()), "results", "Ergebnis_Spurmitte_2026-11-15")
P = os.path.join(ROOT, "parts")
OUT = os.environ.get("BFLC_OUT") or os.path.join(ROOT, "Spurmitte_Ergebnisbericht_2026-11-15.html")
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
            try:                                           # mail size: JPEG quality 68, optimised, progressive (1100 x 760 stays)
                import io
                from PIL import Image
                buf = io.BytesIO()
                Image.open(io.BytesIO(raw)).convert("RGB").save(buf, "JPEG", quality=68, optimize=True, progressive=True, subsampling=2)
                raw = buf.getvalue()
            except Exception:
                pass
            b = base64.b64encode(raw).decode()
            return 'src="data:image/jpeg;base64,%s"' % b
        return m.group(0)
    html = re.sub(r'src="([^"]+\.(?:jpg|jpeg))"', img, html)
    html = html.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    return html


svg = read("02_architecture.svg")
svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg)
sections = [
    ("start", "Start here", render(read("00a_start.md"))),
    ("summary", "Summary", render(read("00_summary.md"), 0)),
    ("gt", "1. Ground truth: what is needed, why, how it is made, how to rebuild it", render(read("01_gt.md"))),
    ("arch", "2. System architecture", '<div class="fig">%s</div>' % svg + render(read("02_architecture.md"))),
    ("results", "3. Experiments, results, compute, difficult situations, own lane", render(read("03_results.md"))),
    ("qual", "4. Qualitative evaluation", render(read("04_qualitative.md"))),
    ("prior", "5. The number of lanes as a prior from the map: pros and cons", render(read("06_lane_count_prior.md"))),
    ("code", "6. Champion code", render(read("07_code.md"))),
    ("open", "7. Open decisions and next steps", render(read("08_open.md"))),
    ("rebuild", "8. Rebuilding it: from nuScenes to the numbers", render(read("09_rebuild.md"))),
]
nav = " | ".join('<a href="#%s">%s</a>' % (k, t.split(".")[0] if t[0].isdigit() else t) for k, t, _ in sections)
body = "\n".join('<section id="%s"><h2>%s</h2>%s</section>' % (k, t, h) for k, t, h in sections if h.strip())
css = """
:root{--bg:#fbfaf7;--fg:#1d2330;--mut:#5b6475;--ln:#d9d6cc;--ac:#1f5fa8;--bgc:#f1efe8}
@media (prefers-color-scheme:dark){:root{--bg:#161a22;--fg:#e6e8ee;--mut:#9aa3b5;--ln:#2d3443;--ac:#6aa7ee;--bgc:#1e2330}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif;margin:0}
main{max-width:980px;margin:0 auto;padding:18px 16px 60px}
h1{font-size:24px;margin:.2em 0}h2{font-size:19px;margin:1.6em 0 .5em;border-bottom:1px solid var(--ln);padding-bottom:4px}
h3{font-size:16px;margin:1.2em 0 .3em}h4{font-size:14px;margin:1em 0 .2em}
code{background:var(--bgc);padding:1px 4px;border-radius:3px;font-size:.88em}
nav{font-size:13px;color:var(--mut);margin:6px 0 14px}nav a{color:var(--ac);text-decoration:none}
.tw{overflow-x:auto;margin:.6em 0}table{border-collapse:collapse;font-size:12.5px}th,td{border:1px solid var(--ln);padding:3px 6px;vertical-align:top}th{background:var(--bgc);text-align:left}
img{max-width:100%;height:auto;border:1px solid var(--ln)}.fig svg{max-width:100%;height:auto;background:#fbfaf7}
p,li{max-width:80ch}.sub{color:var(--mut);font-size:13px}
"""
html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Lane graph results 2026-11-15</title><style>%s</style></head><body><main>'
        '<h1>Lane-centre graph from camera + radar: results, 2026-11-15</h1>'
        '<div class="sub">Spurmitte project. Numbers come from stored cards and result files; VAL = 873 frames / 22 scenes, TEST = 833 frames / 21 scenes (one use, E312 only).</div>'
        '<nav>%s</nav>%s</main></body></html>') % (css, nav, body)
open(OUT, "w", encoding="utf-8").write(html)
print(OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB")
