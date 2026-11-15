s = open("HANDOVER.md", encoding="utf-8").read()
a = s.index("2. **E324** = E320 refined"); b = s.index("3. **E292**")
new = '''2. **E324** = E320 refined (card v3 after two critic passes; arm s ACCEPTED, `temp/e324/ACCEPTED_s` written 21:15, so day14 runs it FIRST tonight, 3.2 h): CPU diagnosis done (`evaluation/jdup.py`, six read-outs + E316 nogeom): the guard excess of E320 on seed 0 is three comparable steps (read-out choice nogeom +0.0062, recipe +0.0064, attached ends +0.0058), on seed 1 mostly the read-out choice (+0.0053 of +0.0093); extra false curves +788 / +352 (E316: -716 / -1219); connector-over-connector overlaps are a MINORITY (43-47 % of the connector duplicate points; 31 % of the excess on seed 0, none on seed 1), so the connector-level Matrix-NMS layer (arm n) is only a stub. Arm s = E320 recipe at 6000 pair steps, read against E316 snap-nogeom (dup 0.1744 / 0.1585): additive prediction dup 0.1866 / 0.1626, topo F1 0.6879 / 0.6985; interaction hypothesis = the excess over E316 falls below +0.0122 / +0.0040. Then ONE stack card on the recipes that already ran.
'''
s = s[:a] + new + s[b:]
open("HANDOVER.md", "w", encoding="utf-8").write(s); print("ok")
