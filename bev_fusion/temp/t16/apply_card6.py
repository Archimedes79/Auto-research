"""scratch: put temp/t16/card_section6.md into the t16 card as section 6 and open a section 7 for the summary (idempotent)"""
p = "bev_fusion/experiments/t16_full_network/experiment.md"
s = open(p, encoding="utf-8").read()
new = open("bev_fusion/temp/t16/card_section6.md", encoding="utf-8").read()
if "## 6 Bring-up log" in s:
    i = s.index("## 6 Bring-up log")
else:
    i = s.index("## 6 Summary (written after the runs)")
s = s[:i] + new.rstrip("\n") + "\n\n## 7 Summary (written after the large run)\n(open)\n"
open(p, "w", encoding="utf-8").write(s)
print("card section 6 written,", len(new), "characters")
