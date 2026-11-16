import base64, io, re
from PIL import Image
a = open("/mnt/c/Auto-research/bev_fusion_lane_center/temp/e330/html_backup/Spurmitte_Ergebnisbericht_2026-11-15.orig.html", encoding="utf-8").read()
b = open("/mnt/c/Auto-research/bev_fusion_lane_center/results/Ergebnis_Spurmitte_2026-11-15/Spurmitte_Ergebnisbericht_2026-11-15.html", encoding="utf-8").read()
strip = lambda t: re.sub(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", "IMG", t)
print("text identical apart from the images:", strip(a) == strip(b), "| images", len(re.findall("data:image/jpeg", a)), len(re.findall("data:image/jpeg", b)))
m = re.findall(r"data:image/jpeg;base64,([A-Za-z0-9+/=]+)", b)[4]
im = Image.open(io.BytesIO(base64.b64decode(m)))
print(im.size)
im.save("/mnt/c/Auto-research/bev_fusion_lane_center/temp/e330/sample_after.png")
