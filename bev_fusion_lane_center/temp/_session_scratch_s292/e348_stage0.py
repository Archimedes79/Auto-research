p = "experiments/E348_e324s-recipe-without-paint-branch/experiment.md"
s = open(p, encoding="utf-8").read()
a = 'so the ledger notes say "by hand: replay".'
assert s.count(a) == 1
b = (a + " **Done 2026-11-15 01:22 (`temp/e348/stage0.sh`, logs `temp/e348/stage0_*.log`):** E324 s unsnapped topo F1 0.6594 / 0.6621 (dup_frac 0.1640 / 0.1385), attach-only 0.6616 / 0.6646 (dup_frac 0.1708 / 0.1399), next to the stored PRIMARY 0.6912 / 0.6948 and snapped 0.6981 / 0.7010: the nearest-piece wiring is worth about +0.030 on the attached ends (attach-only to PRIMARY) and +0.034 / +0.035 from unsnapped to snapped. The expected absolute value of the arm on the attach read-out (prediction -0.005) is 0.6566 / 0.6596.")
s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
print("ok")
