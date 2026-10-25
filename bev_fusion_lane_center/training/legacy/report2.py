"""Berichtswerkzeug (Schleife 2, Englisch): je Experiment eine Seite als JSON + Markdown + selbststaendiges HTML (Bilder eingebettet); der Abschlussbericht fuegt die JSON-Dateien zusammen.
make_report2(exp_id, title, question, setup, tables=[(caption, head, rows)], images=[(path, caption)], findings, limits, arch=None, status='completed')
arch: Funktion ax -> None (Blockdiagramm, siehe report_lib.blocks_arch / rows_arch) oder Pfad zu einem PNG.
Ausgabe: reports/loop2/<exp_id>.{json,md,html} und assets/."""
import base64
import html
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "/mnt/c/ML-Experiment/bev_fusion_lane_center/reports/loop2"

CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;max-width:1050px;margin:0 auto;padding:24px 28px;color:#202124;line-height:1.5;font-size:14.5px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;border-bottom:2px solid #1a73e8;padding-bottom:3px;margin-top:30px}
.sub{color:#5f6368;margin-bottom:14px}.q{background:#e8f0fe;border-left:5px solid #1a73e8;padding:10px 14px;margin:12px 0;border-radius:4px}
table{border-collapse:collapse;width:100%;margin:8px 0 14px;font-size:13px}th,td{border:1px solid #dadce0;padding:4px 8px;text-align:center}
th{background:#f1f3f4}td:first-child,th:first-child{text-align:left}tr:nth-child(even) td{background:#fafafa}
figure{margin:10px 0 16px;text-align:center;page-break-inside:avoid}figure img{max-width:100%;border:1px solid #e0e0e0}
figcaption{font-size:12.5px;color:#5f6368;margin-top:4px;text-align:left}.cap{font-weight:600;margin:12px 0 2px}
.wide td:nth-child(2){min-width:230px;text-align:left}.wide td:nth-child(1){min-width:70px}ul{margin:4px 0 8px 0;padding-left:22px}code{background:#f1f3f4;padding:1px 5px;border-radius:3px;font-size:12.5px}
@media print{body{max-width:none;padding:0 6mm;font-size:12px}}
"""


def b64(path):
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    return f"data:image/{ext};base64," + base64.b64encode(open(path, "rb").read()).decode()


def f3(x, nd=3):
    try:
        return "-" if x is None or x != x else f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def table_html(head, rows):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    r = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table class='{'wide' if len(head) > 9 else ''}'><tr>{h}</tr>{r}</table>"


def ul(items):
    return "<ul>" + "".join(f"<li>{html.escape(str(i))}</li>" for i in items) + "</ul>"


def render_html(d, base_dir):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(d['exp_id'])} {html.escape(d['title'])}</title><style>{CSS}</style></head><body>"]
    parts.append(f"<h1>{html.escape(d['exp_id'])}: {html.escape(d['title'])}</h1><div class='sub'>Status: {html.escape(d['status'])}</div>")
    parts.append(f"<div class='q'><b>Hypothesis / question.</b> {html.escape(d['question'])}</div>")
    parts.append("<h2>Setup</h2>" + ul(d["setup"]))
    if d.get("architektur"):
        p = f"{base_dir}/{d['architektur']}"
        if os.path.exists(p):
            parts.append(f"<figure><img src='{b64(p)}'><figcaption>Architecture / data flow of this experiment.</figcaption></figure>")
    parts.append("<h2>Numbers</h2>")
    for cap, head, rows in d["tables"]:
        parts.append(f"<div class='cap'>{html.escape(cap)}</div>" + table_html(head, rows))
    parts.append("<h2>Result images</h2>")
    for im in d["images"]:
        p = f"{base_dir}/{im['rel']}"
        if os.path.exists(p):
            parts.append(f"<figure><img src='{b64(p)}'><figcaption>{html.escape(im['cap'])}</figcaption></figure>")
    parts.append("<h2>Findings (qualitative and quantitative)</h2>" + ul(d["findings"]))
    if d["limits"]:
        parts.append("<h2>Limits and caveats</h2>" + ul(d["limits"]))
    parts.append("</body></html>")
    return "\n".join(parts)


def make_report2(exp_id, title, question, setup, tables, images, findings, limits=(), arch=None, status="completed", out_dir=OUT, arch_height=3.0):
    os.makedirs(f"{out_dir}/assets", exist_ok=True)
    img = []
    for p, cap in images:
        if not os.path.exists(p):
            print("Bild fehlt:", p)
            continue
        dst = f"{out_dir}/assets/{exp_id}_{os.path.basename(p)}"
        shutil.copyfile(p, dst)
        img.append(dict(rel=f"assets/{exp_id}_{os.path.basename(p)}", cap=cap))
    archrel = None
    if arch is not None:
        archrel = f"assets/{exp_id}_architecture.png"
        if callable(arch):
            fa, axa = plt.subplots(figsize=(15, arch_height))
            arch(axa)
            fa.tight_layout()
            fa.savefig(f"{out_dir}/{archrel}", dpi=110)
            plt.close(fa)
        else:
            shutil.copyfile(arch, f"{out_dir}/{archrel}")
    d = dict(exp_id=exp_id, title=title, status=status, question=question, setup=list(setup), tables=[(c, list(h), [list(map(str, r)) for r in rows]) for c, h, rows in tables],
             images=img, architektur=archrel, findings=list(findings), limits=list(limits))
    json.dump(d, open(f"{out_dir}/{exp_id}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    md = [f"# {exp_id}: {title}", "", f"Status: {status}", "", f"**Question:** {question}", "", "## Setup"] + [f"- {a}" for a in setup]
    if archrel:
        md += ["", f"![architecture]({archrel})"]
    md += ["", "## Numbers"]
    for cap, head, rows in d["tables"]:
        md += ["", f"**{cap}**", "", "| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"] + ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "## Result images", ""]
    for im in img:
        md += [f"![{im['cap']}]({im['rel']})", f"*{im['cap']}*", ""]
    md += ["## Findings"] + [f"- {a}" for a in findings]
    if limits:
        md += ["", "## Limits"] + [f"- {a}" for a in limits]
    open(f"{out_dir}/{exp_id}.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(f"{out_dir}/{exp_id}.html", "w", encoding="utf-8").write(render_html(d, out_dir))
    return f"{out_dir}/{exp_id}.html"
