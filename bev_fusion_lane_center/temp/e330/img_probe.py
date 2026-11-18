import re, base64, io, sys
from PIL import Image
s = open("/mnt/c/Auto-research/bev_fusion_lane_center/results/Ergebnis_Spurmitte_2026-11-15/Spurmitte_Ergebnisbericht_2026-11-15.html", encoding="utf-8").read()
print("html chars", len(s), "bytes", len(s.encode("utf-8")))
for m in re.finditer(r'data:image/jpeg;base64,([A-Za-z0-9+/=]+)', s):
    b = base64.b64decode(m.group(1))
    im = Image.open(io.BytesIO(b))
    print(len(b), im.size, im.mode)
