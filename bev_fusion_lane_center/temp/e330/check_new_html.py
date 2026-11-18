import re, html
s = open("/mnt/c/Auto-research/bev_fusion_lane_center/results/Ergebnis_Spurmitte_2026-11-15/Spurmitte_Ergebnisbericht_2026-11-15.html", encoding="utf-8").read()
print("bytes", len(s.encode()), "| images", s.count("data:image/jpeg"), "| tables", s.count("<table>"), "| divs open/close", s.count("<div"), s.count("</div>"))
print(re.findall(r'<nav>(.*?)</nav>', s)[0][:400])
for h in re.findall(r"<h[23][^>]*>([^<]*)", s):
    if re.match(r"(1\.|8\.)", h.strip()) or h.strip().startswith(("1 ", "8 ")): print(" ", h.strip())
txt = re.sub(r"<[^>]+>", "", s)
for bad in ["\|", "&amp;lt;", "**", "```"]:
    print("raw", repr(bad), txt.count(bad))
i = txt.find("What the GT must contain")
print(html.unescape(txt[i:i+500]))
