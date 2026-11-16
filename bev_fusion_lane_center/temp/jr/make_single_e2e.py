"""one self-contained html of the E2E report: pictures AND videos embedded (base64). usage: python3 make_single_e2e.py  (reads results/E2E.report/index.html, writes Spurmitte_E2E_report_<date>.html next to it)"""
import base64, os, re, sys, time
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "E2E.report")
s = open(os.path.join(R, "index.html"), encoding="utf-8").read()
def b64(path, mime):
    return f"data:{mime};base64," + base64.b64encode(open(os.path.join(R, path), "rb").read()).decode()
s = re.sub(r'(src|href)="(img/[^"]+)"', lambda m: f'{m.group(1)}="{b64(m.group(2), "image/jpeg")}"', s)
s = re.sub(r'(src)="(vid/[^"]+)"', lambda m: f'{m.group(1)}="{b64(m.group(2), "video/webm")}"', s)
out = os.path.join(R, "Spurmitte_E2E_report_" + time.strftime("%Y-%m-%d") + ".html")
open(out, "w", encoding="utf-8").write(s)
print(out, round(len(s) / 1e6, 2), "MB")
