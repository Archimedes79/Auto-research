p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
i = s.index("- **E283 (analysis card")
j = s.index("## Open for Mirko")
new = ("- **E283 (analysis card, CPU only, sections 1 - 7):** junction blur = unknown entry -> exit PAIRS; largest graph loss = local 1 - 2.5 m offset in the head's own map; five CPU levers closed. D71 (section 5): the map's end point is about where the paint ends, the defect is the END FLAG (missing at 25 - 33 % of GT ends, ~half of the flags at no GT end). Direction ~99 % right. Scorer oracle (section 7) NOT PASSED on junction F1 (+0.029 / +0.049, needed +0.03) but topo F1 +0.10, lanes found +0.22, Q +0.07 with clean ends and true pairs; with classifier pairs about +0.04.\n\n")
open(p, "w", encoding="utf-8").write(s[:i] + new + s[j:])
