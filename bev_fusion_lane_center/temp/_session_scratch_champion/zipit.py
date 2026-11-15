import os, zipfile
out = "results/Ergebnis_Spurmitte_2026-11-15/champion_code.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, dirs, files in os.walk("champion"):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            if f.endswith((".py", ".md")):
                z.write(os.path.join(d, f))
print(out, os.path.getsize(out), "bytes")
print("\n".join(zipfile.ZipFile(out).namelist()))
