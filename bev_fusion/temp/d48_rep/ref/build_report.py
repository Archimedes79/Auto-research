"""scratch of the t16 report (D46): one self-contained HTML report from the stored files of the runs (numbers from the evaluation tables, never typed), re-runnable as more runs finish.
    python3 bev_fusion/temp/t16/build_report.py [--out bev_fusion/results/t16/report.html]
Runs (missing ones are skipped): t16_full_s0 (baseline), t16_ctl_s0 (control), t17_data_s0 (more data), t18_fix_s0 (bundle). Run in the gcar env (PIL)."""
import argparse
import base64
import html
import io
import json
import os
import re
import subprocess

ROOT = "bev_fusion/runs"
RUNS = [("t16_full_s0", "baseline (full network)"), ("t16_ctl_s0", "control (old chain)"), ("t17_data", "more data (700 scenes)"), ("t18_fix", "bundle (more data + fixes + M lr)")]


def table(run, n=1000):
    d = {}
    p = f"{ROOT}/{run}/eval/by_n/table_n{n}.txt"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            m = re.match(r"^(\S+)\s+(-?[0-9.]+(?:e-?[0-9]+)?|nan)\s*$", line.rstrip())
            if m:
                d[m.group(1)] = float(m.group(2))
    return d


def esc(s):
    return html.escape(str(s))


def img(path, w=1100):
    from PIL import Image
    if not os.path.exists(path):
        return ""
    im = Image.open(path).convert("RGB")
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=82)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(b.getvalue()).decode()}" alt="{esc(os.path.basename(path))}">'


def video(path):
    if not os.path.exists(path) or os.path.getsize(path) > 12e6:
        return f"<p><em>video {esc(os.path.basename(path))}: see the file next to this report</em></p>"
    return (f'<video controls preload="none" src="data:video/mp4;base64,{base64.b64encode(open(path, "rb").read()).decode()}"></video>'
            f"<div class=cap>{esc(os.path.basename(path))}</div>")


def rows_table(spec, tabs, labels, fmt="{:.3f}"):
    out = ["<table><tr><th>metric</th>" + "".join(f"<th>{esc(l)}</th>" for l in labels) + "</tr>"]
    for name, key, lower in spec:
        vals = [t.get(key) for t in tabs]
        cells = []
        for i, v in enumerate(vals):
            if v is None:
                cells.append("<td>-</td>")
                continue
            c = fmt.format(v)
            if i > 0 and vals[0] is not None:
                d = v - vals[0]
                good = d < 0 if lower else d > 0
                cls = "" if abs(d) < 0.005 else (" good" if good else " bad")
                c += f' <span class="delta{cls}">{d:+.3f}</span>'
            cells.append(f"<td>{c}</td>")
        out.append(f"<tr><td>{esc(name)}</td>" + "".join(cells) + "</tr>")
    return "\n".join(out) + "</table>"


MAPS = [("drivable area IoU", "seg_fov/seg/dt+0.00/drivable_area/all/all/iou@0", False), ("walkway IoU", "seg_fov/seg/dt+0.00/walkway/all/all/iou@0", False),
        ("car park IoU", "seg_fov/seg/dt+0.00/carpark_area/all/all/iou@0", False), ("pedestrian crossing IoU", "seg_fov/seg/dt+0.00/ped_crossing/all/all/iou@0", False),
        ("lane divider IoU (0.5 m)", "seg_fov/seg/dt+0.00/lane_divider/all/all/iou@0.5", False), ("road divider IoU (1 m)", "seg_fov/seg/dt+0.00/road_divider/all/all/iou@1", False),
        ("drivable IoU at dt -0.5 s", "seg_fov/seg/dt-0.50/drivable_area/all/all/iou@0", False), ("drivable IoU at dt +0.5 s", "seg_fov/seg/dt+0.50/drivable_area/all/all/iou@0", False),
        ("drivable ROC-AUC", "curve/dt+0.00/drivable_area/roc_auc", False), ("walkway AP", "curve/dt+0.00/walkway/ap", False), ("lane divider AP", "curve/dt+0.00/lane_divider/ap", False)]
OBJ = [("mAP (fusion area)", "detf/dt+0.00/mean/all/mAP", False), ("vehicle AP@2 m", "detf/dt+0.00/vehicle/all/ap@2", False), ("pedestrian AP@2 m", "detf/dt+0.00/pedestrian/all/ap@2", False),
       ("two-wheeler AP@2 m", "detf/dt+0.00/two_wheeler/all/ap@2", False), ("vehicle position error ate (m)", "detf/dt+0.00/vehicle/all/ate", True),
       ("vehicle footprint IoU (dense output)", "seg_obj/seg/dt+0.00/vehicle/all/all/iou@0", False)]
VEL = [("velocity error at moving box centres (m/s)", "vel/dt+0.00/centre/field/moving/epe", True), ("same, zero predictor", "vel/dt+0.00/centre/zero/moving/epe", True),
       ("error on moving vehicle cells (m/s)", "vel/dt+0.00/cells/vehicle_moving/epe", True), ("error on moving pedestrian cells (m/s)", "vel/dt+0.00/cells/pedestrian_moving/epe", True),
       ("error on static cells (m/s)", "vel/dt+0.00/cells/static/epe", True)]

COV = [(f"{cls} recall@2 m, {lab}", f"cov/dt+0.00/{cls}/{k}/recall2m", False) for cls in ("vehicle", "pedestrian") for lab, k in (("camera + radar", "cam_radar"), ("camera only", "cam_only"), ("radar only", "radar_only"), ("neither sensor", "neither"))]

CSS = """
:root{--bg:#fff;--fg:#1d2330;--mut:#5b6577;--line:#dde2ea;--acc:#1f5fbf;--good:#16794a;--bad:#b3261e;--card:#f6f8fb}
@media (prefers-color-scheme:dark){:root{--bg:#12161d;--fg:#e6e9ef;--mut:#9aa4b5;--line:#2b3340;--acc:#6aa3ff;--good:#4cc38a;--bad:#ff8a80;--card:#1a2029}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:.2em 0}h2{font-size:20px;margin:1.6em 0 .4em;border-top:1px solid var(--line);padding-top:.8em}h3{font-size:16px;margin:1.2em 0 .3em}
.lead{color:var(--mut)}table{border-collapse:collapse;width:100%;margin:.6em 0;font-size:14px}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--mut);font-weight:600}img,video{max-width:100%;border:1px solid var(--line);border-radius:6px;margin:.4em 0}
.delta{font-size:12px;color:var(--mut)}.delta.good{color:var(--good)}.delta.bad{color:var(--bad)}.cap{color:var(--mut);font-size:13px;margin-bottom:1em}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:.8em 0}code{background:var(--card);padding:1px 4px;border-radius:3px;font-size:13px}
ul{padding-left:1.2em}li{margin:.25em 0}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bev_fusion/results/t16/report.html")
    ap.add_argument("--text", default="bev_fusion/temp/t16/report_text.py")
    a = ap.parse_args()
    runs = [(r, l) for r, l in RUNS if table(r)]
    tabs = [table(r) for r, _ in runs]
    labels = [l for _, l in runs]
    import importlib.util
    spec = importlib.util.spec_from_file_location("report_text", a.text)
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    sec = {"title": rt.TITLE, "lead": rt.LEAD, "blocks": rt.BLOCKS}
    head_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = [f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>{esc(sec['title'])}</title><style>{CSS}</style><main>",
           f"<h1>{esc(sec['title'])}</h1><p class=lead>{sec['lead']} Repository state: <code>{head_commit}</code>. Numbers come from the stored evaluation files of the runs (VAL, 1000 frames, GT v8 (store <code>c643a7539d</code>, the trainer's GT cache key <code>0.5_4_6a7aa7864a</code>, identical in every run of this report)), never typed by hand; every comparison is single seed unless it says otherwise.</p>"]
    for heading, body in sec["blocks"]:
        out.append(f"<h2>{esc(heading)}</h2>{body}")
        if heading.startswith("Results") and tabs:
            out.append("<h3>Maps (VAL, fusion-area scoring)</h3>" + rows_table(MAPS, tabs, labels))
            out.append("<h3>Objects</h3>" + rows_table(OBJ, tabs, labels))
            out.append("<h3>Velocity (lower is better)</h3>" + rows_table(VEL, tabs, labels))
            out.append("<h3>Detection recall by the sensors that see the object (does the radar help?)</h3>" + rows_table(COV, tabs, labels))
            for r, l in runs[:1]:
                for f in ("scene-0103_020_overview.png", "scene-0039_026_overview.png"):
                    p = f"{ROOT}/{r}/eval/images/{f}"
                    if os.path.exists(p):
                        out.append(img(p) + f"<div class=cap>{esc(l)}: {esc(f)} (camera with GT, GT, prediction with boxes, error maps)</div>")
                for k in (0, 2):
                    p = f"{ROOT}/{r}/eval/modules/frame_{k}.png"
                    if os.path.exists(p):
                        out.append(img(p) + f"<div class=cap>{esc(l)}: every stage of the chain on one validation frame (module probe)</div>")
                vd = f"{ROOT}/{r}/eval/suite/qualitative/videos"
                if os.path.isdir(vd):
                    for f in sorted(os.listdir(vd)):
                        if f.endswith("_dashboard.mp4"):
                            out.append(video(os.path.join(vd, f)))
    out.append("</main></html>")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(out))
    print("written", a.out, f"{os.path.getsize(a.out) / 1e6:.1f} MB", "runs:", [r for r, _ in runs])


if __name__ == "__main__":
    main()
