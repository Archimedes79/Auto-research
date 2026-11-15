p = "experiments/E343_test-e324s/experiment.md"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("- **E312's per-scene counts (D95 reading, NOT yet Mirko's):**", "- **E312's per-scene counts (D95 reading R1, CONFIRMED by Mirko, see below):**")
rep("The three readings of D95 that this card makes and that Mirko has not confirmed (`D95.md` says \"my reading, which he can correct\" and has no \"Confirmed\" line): (R1) the per-scene recompute of E312; (R2) arm p and snap-nogeom are part of \"E324 s\"; (R3) arm k is \"(+ c2)\" and \"the stack E338\". They go to Mirko in the next status message. **Until a Confirmed line for R1 exists the STRICT reading is the default: no recompute, the fallback of section 0 (no interval).** If it is confirmed, the recompute is used only if it reproduces `card_s<S>.json` to <= 1e-9 in every number on that seed (and then on both).",
    "The three readings of D95 that this card makes: (R1) the per-scene recompute of E312; (R2) arm p and snap-nogeom are part of \"E324 s\"; (R3) arm k is \"(+ c2)\" and \"the stack E338\". **Confirmed by Mirko, 2026-11-14 about 22:25, in chat (\"mach alles\"), recorded as the Confirmed line of `D95.md` (commit e87fb70b):** R1, R2 and R3 stand as this card words them; the strict reading (no recompute) no longer applies. The recompute is used only if it reproduces `card_s<S>.json` to <= 1e-9 in every number on that seed (and then on both); otherwise the fallback of section 0 (no interval) applies to both seeds.")
rep("(`--control-ends-dump` is used only after a Confirmed line for R1; otherwise the strict reading drops it.)", "(`--control-ends-dump` is used: R1 is confirmed; the E321 dump is read only, its sha256 is checked at the start and the end.)")
rep("a note that starts with \"Mirko's OK, once (D95, 2026-11-14, \"2. ja\")\",", "a note that starts with \"Mirko's OK, once (D95, 2026-11-14, \"2. ja\"; the three readings confirmed 2026-11-14 about 22:25 with \"mach alles\")\",")
open(p, "w", encoding="utf-8").write(s)
print("ok")
