"""One-off (report of experiment t1): for every validation frame that both runs rendered (same file name in runs/<run>/eval/images),
one PNG with the baseline's panel on top and the combined system's below, with a label bar.
Usage: python temp/train/stack_images.py runs/t1_r0 runs/t1_r3_combined experiments/t1_full_chain/images"""
import os
import sys

from PIL import Image, ImageDraw

a, b, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
da, db = os.path.join(a, "eval", "images"), os.path.join(b, "eval", "images")
names = sorted(set(os.listdir(da)) & set(os.listdir(db)))
for n in names:
    ia, ib = Image.open(os.path.join(da, n)).convert("RGB"), Image.open(os.path.join(db, n)).convert("RGB")
    w = max(ia.width, ib.width)
    bar = 34
    canvas = Image.new("RGB", (w, ia.height + ib.height + 2 * bar), "white")
    d = ImageDraw.Draw(canvas)
    d.text((10, 10), f"BASELINE {os.path.basename(a)}: window-only camera memory, plain heading head, 5000 steps", fill=(0, 0, 0))
    canvas.paste(ia, (0, bar))
    d.text((10, bar + ia.height + 10), f"COMBINED {os.path.basename(b)}: carried world memory, direction head, S5c box refinement, 6000 steps", fill=(0, 0, 0))
    canvas.paste(ib, (0, 2 * bar + ia.height))
    canvas.save(os.path.join(out, "compare_" + n))
    print("wrote", os.path.join(out, "compare_" + n))
