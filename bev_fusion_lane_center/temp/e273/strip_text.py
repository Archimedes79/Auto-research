import re
import sys

src = sys.argv[1]
h = open(src, encoding="utf-8").read()
h = re.sub(r"<svg.*?</svg>", "[SVG]", h, flags=re.S)
h = re.sub(r"<style.*?</style>", "", h, flags=re.S)
h = re.sub(r"<script.*?</script>", "", h, flags=re.S)
t = re.sub(r"<(h[1-3])[^>]*>", r"\n\n### ", h)
t = re.sub(r"<(p|li|figcaption|tr|details|summary|video)[^>]*>", r"\n", t)
t = re.sub(r"<[^>]+>", "", t)
t = re.sub(r"\n\s*\n+", "\n", t)
open(sys.argv[2], "w", encoding="utf-8").write(t)
print(len(t))
