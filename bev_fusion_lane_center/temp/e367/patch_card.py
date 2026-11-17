p = "experiments/E367_e365-capacity-balance-schedule-arms/experiment.md"
s = open(p, encoding="utf-8").read()
a = "**Expected tie to E366:**"
assert s.count(a) == 1
s = s.replace(a, "**What the none weight changes, said before the runs (code critic of the options, 2026-11-17 01:30):** W is an ABSOLUTE down-weight of the empty-G source terms (they are still divided by the number of sources, not by the sum of the weights), and it also weights the none class of the type cross-entropy (normalised by the sum of the weights); so b_bal moves TWO things at once: the none slot of the wiring and, through the type CE, the key weight 1 - p_none of the soft pruning; the reading of the b arms names both. " + a, 1)
b = "**Note on the JSON:**"
assert s.count(b) == 1
s = s.replace(b, "**Recipe note:** `--ffn` is passed only for a_wide (192); the other arms leave it at its default (2 x d); a resume with another `--ffn` spelling is refused by the recipe check (conservative). Token-drop together with a none weight (not used here) would turn a source whose only successor was dropped into an empty-G source that takes W: recorded now, before that combination is ever run. **Start:** b_bal5 pair started 2026-11-17 01:33 (system clock) from the trainer the code critic accepted. " + b, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
