"""one self-contained html of the junction report: images inline (base64), videos replaced by a note with the file name (the webm files are sent next to it). usage: make_single.py"""
import base64
import os
import re

R = os.path.join(os.path.dirname(__file__), "..", "..", "results", "Junction.report")
s = open(os.path.join(R, "index.html"), encoding="utf-8").read()


def uri(path):
    ext = os.path.splitext(path)[1][1:].replace("jpg", "jpeg")
    return f"data:image/{ext};base64," + base64.b64encode(open(os.path.join(R, path), "rb").read()).decode()


s = re.sub(r'(src|href)="(img/[^"]+)"', lambda m: f'{m.group(1)}="{uri(m.group(2))}"', s)
s = re.sub(r'<video[^>]*src="vid/([^"]+)"[^>]*></video>', lambda m: f'<p class="cap"><b>Video (attached as a separate file): {m.group(1)}</b></p>', s)
open(os.path.join(R, "Junction_report_single.html"), "w", encoding="utf-8").write(s)
print(len(s) / 1e6, "MB")
