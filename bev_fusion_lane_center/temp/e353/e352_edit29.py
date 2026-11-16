import re
p = "evaluation/jprune_b.py"
s = open(p, encoding="utf-8").read()
s = s.replace("GATE_ABS, GATE_REL = 2e-3, 2e-3                                                                                            # every other number: |difference| <= max(GATE_ABS, GATE_REL * |stored|)",
              "GATE_ABS, GATE_REL = 2e-3, 1e-3                                                                                            # every other number: |difference| <= max(GATE_ABS, GATE_REL * |stored|) (observed, 2 seeds x 3 read-outs x base / head: at most 8.6e-4 on the 0 - 1 numbers = 0.43 of GATE_ABS, 4.5e-3 on vert100_dp (about 19) = 0.23 of GATE_REL * 19.6)")
s = s.replace("every other number within the tolerances of `gate`", "every other number within the tolerances of `gate`")
s = s.replace("topo F1 within 2e-4, every other number within max(2e-3, 2e-3 x stored)", "topo F1 within 2e-4, every other number within max(2e-3, 1e-3 x stored)")
open(p, "w", encoding="utf-8").write(s)
for p in ("temp/day30/day30.sh", "evaluation/README.md", "temp/e353/build_notes.md", "evaluation/tests/test_jprune_b.py"):
    t = open(p, encoding="utf-8").read()
    t = t.replace("max(2e-3, 2e-3 x stored)", "max(2e-3, 1e-3 x stored)").replace("max(2e-3, 2e-3 * 19)", "max(2e-3, 1e-3 * 19)").replace("within max(2e-3, 2e-3 x stored)", "within max(2e-3, 1e-3 x stored)")
    open(p, "w", encoding="utf-8", newline="\n" if p.endswith(".sh") else None).write(t)
