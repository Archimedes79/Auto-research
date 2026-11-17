"""scratch: PNG -> JPEG (width <= 1100, quality 80, lowered until <= 160 KB)"""
import os, sys, glob
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
tot = 0
for p in sorted(glob.glob(os.path.join(src, "*.png"))):
    im = Image.open(p).convert("RGB")
    if im.width > 1100:
        im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
    out = os.path.join(dst, os.path.basename(p)[:-4] + ".jpg")
    for q in (80, 76, 72, 68, 64):
        im.save(out, "JPEG", quality=q, optimize=True, progressive=True)
        if os.path.getsize(out) <= 160 * 1024:
            break
    tot += os.path.getsize(out)
    print(os.path.basename(out), im.size, q, os.path.getsize(out))
print("total", tot)
