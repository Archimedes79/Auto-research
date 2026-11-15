import re
import sys
import zipfile

z = zipfile.ZipFile(sys.argv[1])
slides = sorted(
    (n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
    key=lambda n: int(re.findall(r"\d+", n)[0]),
)
print("slides:", len(slides))
for n in slides:
    x = z.read(n).decode("utf8")
    paras = re.findall(r"<a:p>(.*?)</a:p>", x, flags=re.S)
    print("=== " + n)
    for p in paras:
        t = "".join(re.findall(r"<a:t>(.*?)</a:t>", p, flags=re.S))
        if t.strip():
            print("  " + t)
