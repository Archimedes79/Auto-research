"""One-off (t3 report): mosaic of the same val frames for four systems: camera | GT | camera only | radar only | camera + radar | long-trained.
Crops the panels of the evaluate_run images (fixed layout) of runs/t3_img_*."""
import glob
import os
from PIL import Image, ImageDraw

RUNS = [("camera only (3000 steps)", "t3_img_cam"), ("radar only (3000 steps)", "t3_img_rad"),
        ("camera + radar (3000 steps)", "t3_img_ref"), ("camera + radar (7000 x 2 steps)", "t3_img_long")]
files = sorted(os.path.basename(f) for f in glob.glob("runs/t3_img_ref/eval/images/*.png"))
pick = [files[i] for i in (1, 2, 4, 6)]
REF = 1350.0
def crop(im, box):
    s = im.width / REF
    return im.crop(tuple(int(v * s) for v in box))
CAM, GT, PRED = (20, 125, 440, 380), (485, 55, 845, 470), (925, 55, 1285, 470)
rows = []
for f in pick:
    base = Image.open(f"runs/t3_img_ref/eval/images/{f}").convert("RGB")
    cells = [("camera", crop(base, CAM).resize((420, 255))), ("ground truth", crop(base, GT).resize((330, 380)))]
    for label, run in RUNS:
        im = Image.open(f"runs/{run}/eval/images/{f}").convert("RGB")
        cells.append((label, crop(im, PRED).resize((330, 380))))
    H = 380 + 28
    row = Image.new("RGB", (sum(c.width for _, c in cells), H), "white")
    x = 0
    d = ImageDraw.Draw(row)
    for label, c in cells:
        row.paste(c, (x, 28 + (380 - c.height) // 2))
        d.text((x + 4, 6), label, fill=(20, 20, 20))
        x += c.width
    d.text((x - 130, 6), f.split("_", 1)[1][:-4], fill=(90, 90, 90))
    rows.append(row)
W = max(r.width for r in rows)
out = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
y = 0
for r in rows:
    out.paste(r, (0, y))
    y += r.height
os.makedirs("experiments/t3_improve_loop/images", exist_ok=True)
out.save("experiments/t3_improve_loop/images/compare_systems.png")
print(out.size, pick)
