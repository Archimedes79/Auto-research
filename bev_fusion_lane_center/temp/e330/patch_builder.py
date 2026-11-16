p = "temp/jr/build_ergebnis.py"
s = open(p, encoding="utf-8").read()
s = s.replace('''            b = base64.b64encode(open(f, "rb").read()).decode()''', '''            raw = open(f, "rb").read()
            try:                                           # mail size: JPEG quality 68, optimised, progressive (1100 x 760 stays)
                import io
                from PIL import Image
                buf = io.BytesIO()
                Image.open(io.BytesIO(raw)).convert("RGB").save(buf, "JPEG", quality=68, optimize=True, progressive=True, subsampling=2)
                raw = buf.getvalue()
            except Exception:
                pass
            b = base64.b64encode(raw).decode()''')
s = s.replace('''    ("open", "7. Open decisions and next steps", render(read("08_open.md"))),
]''', '''    ("open", "7. Open decisions and next steps", render(read("08_open.md"))),
    ("rebuild", "8. Rebuilding it: from nuScenes to the numbers", render(read("09_rebuild.md"))),
]''')
open(p, "w", encoding="utf-8").write(s)
p = "results/Ergebnis_Spurmitte_2026-11-15/parts/00_summary.md"
s = open(p, encoding="utf-8").read()
s = s.rstrip("\n") + "\n- **To rebuild it:** chapter 1 says what the ground truth must contain and why, and how it is made from the nuScenes map (recipe, record, checks, limits); chapter 2 the architecture; chapter 8 the data and splits, shapes, training recipe with commands, losses, read-out, metrics and the numbers to hit at each stage.\n"
open(p, "w", encoding="utf-8").write(s)
