import xml.dom.minidom, sys
p = "results/Ergebnis_E2E_Champion_2026-11-17/parts/02_architecture.svg"
xml.dom.minidom.parse(p)
print("svg ok")
h = open("results/Ergebnis_E2E_Champion_2026-11-17/E2E_Champion_Bericht_SKELETON.html", encoding="utf-8").read()
print("sections:", h.count("<section"), "svg in html:", "<svg" in h, "figure refs unresolved:", h.count("02_architecture.svg"))
