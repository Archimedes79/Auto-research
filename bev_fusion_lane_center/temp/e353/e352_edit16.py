p = "temp/e353/build_notes.md"
s = open(p, encoding="utf-8").read()
old = "the null of seed 0 is done (655 s) and its snapped card is in runs/e352_null_s0/ends_card_null_snap_thr0.1.json (the tagged card; both made under the CPU workaround)."
new = "the null of seed 0 was dumped and scored by hand as the build's equality check (a DIAGNOSTIC, not a run.py run: it feeds no verdict); the folder was moved to temp/e352/diag_null_s0/ (with its snapped card ends_card_null_snap_thr0.1.json) so that the run.py job E352.s0_null_s0 can write runs/e352_null_s0/ fresh."
assert old in s
open(p, "w", encoding="utf-8").write(s.replace(old, new))
