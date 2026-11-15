"""Re-encode the embedded JPEGs of the result report (scratch): quality Q, optimised, progressive; same size in pixels; writes the html in place (the original is in temp/e330/html_backup/)."""
import base64, io, re, sys
from PIL import Image
Q = int(sys.argv[1]) if len(sys.argv) > 1 else 68
P = "/mnt/c/Auto-research/bev_fusion_lane_center/results/Ergebnis_Spurmitte_2026-11-15/Spurmitte_Ergebnisbericht_2026-11-15.html"
O = "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e330/html_backup/Spurmitte_Ergebnisbericht_2026-11-15.orig.html"
s = open(O, encoding="utf-8").read()


def re_enc(m):
    im = Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=Q, optimize=True, progressive=True, subsampling=2)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


out = re.sub(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", re_enc, s)
open(P, "w", encoding="utf-8").write(out)
print("quality", Q, "bytes", len(out.encode("utf-8")), "original", len(s.encode("utf-8")))
