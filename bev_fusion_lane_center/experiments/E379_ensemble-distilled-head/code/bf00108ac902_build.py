"""Build the one-file report: the template with every {{IMG:name}} replaced by a data URI, wrapped in a standalone HTML document."""
import base64, os, re, sys
root = os.environ.get("BFLC_ROOT") or os.getcwd()
tpl = open(os.path.join(root, "temp/_session_scratch_s292/report/template.html"), encoding="utf-8").read()
imgdir = os.path.join(root, "results/Gesamtbericht_Spurmitte/img")
def img(m):
    p = os.path.join(imgdir, m.group(1))
    return "data:image/jpeg;base64," + base64.b64encode(open(p, "rb").read()).decode("ascii")
body = re.sub(r"\{\{IMG:([^}]+)\}\}", img, tpl)
assert "{{IMG" not in body
title = re.search(r"<title>(.*?)</title>", body).group(1)
head_end = body.index("</style>") + len("</style>")
head, rest = body[:head_end], body[head_end:]
doc = ('<!doctype html>\n<html lang="de">\n<head>\n<meta charset="utf-8">\n'
       '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head + '\n</head>\n<body>\n' + rest + '\n</body>\n</html>\n')
out = os.path.join(root, "results/Spurmitte_Stand_2026-11-14/Spurmitte_Stand_2026-11-14.html")
open(out, "w", encoding="utf-8", newline="\n").write(doc)
print(out, round(len(doc.encode("utf-8")) / 1e6, 2), "MB")
