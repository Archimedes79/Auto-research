"""scratch: rough text-fit check of the architecture boxes (px per char: title 7.0, line 5.9, iface line 6.3)"""
import re
src = open("/mnt/c/Auto-research/bev_fusion/temp/overview/build.py", encoding="utf-8").read()
ns = {}
code = src.split("# ---------------------------------------------------------------- media as data uris")[0]
exec(code, ns)
bad = 0
for (i, x, y, w, h, title, lines, cls) in ns["boxes"]:
    avail = w - 18
    cw = 6.3 if cls == "iface" else 5.9
    if title and len(title) * 7.0 > avail:
        print("TITLE too wide:", i, title, round(len(title) * 7.0), ">", avail); bad += 1
    for ln in lines:
        if len(ln) * cw > avail:
            print("LINE too wide:", i, ln, round(len(ln) * cw), ">", avail); bad += 1
    n = (1 if title else 0) + len(lines)
    need = 18 + 16 * (1 if title else 0) + 15 * len(lines) if title else 8 + 15 * len(lines) + 10
    if need > h:
        print("HEIGHT:", i, need, ">", h); bad += 1
print("problems:", bad)
