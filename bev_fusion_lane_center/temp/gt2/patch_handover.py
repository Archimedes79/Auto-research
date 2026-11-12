p = "HANDOVER.md"
s = open(p, encoding="utf-8").read()
a = s.index("- **GT merge (D83) in progress:**")
b = s.index("## Next")
new = ("- **GT merge (D83) DONE 2026-11-11 23:05:** `prep/v13-l6` merged into main (68817b0a); the l6 lane store (9ad1d4ce30, 34,149 frames) is in `data/autogt/lane/` next to l4; junction GT identical to l4 on 440 VAL frames (entries 13 of 417 frames up to 0.46 m); new processes read l6, running ones keep l4. BEV chat told. Left over: the worktree `temp/gt_l6` and `temp/gt_l6_store` (scratch duplicates, delete only through `package.py` / `clean_temp.py`), the full `GtGenerator selftest` of the merged tree (run after the night chains), the BEV rig store / front v13 store (BEV chat).\n"
       "- **New since 22:30:** cards E294-E303 (plan `experiment_execution/plans/junction_thread_2026-11-11.md`); E296 (protect special nodes: TIE at p 0.9), E300 (E284 head: far range > 35 m and Singapore are the holes), E301 closed by data (99.8 % of splits / merges are inside junctions), E293 (240 frames: the chord-tangent curve is good only 5 m deep: learned handle scales, card E303), E298 code ready (default off), E286 control c measured (B event heat: entry / exit F1 0.41 / 0.39).\n\n")
s = s[:a] + new + s[b:]
s = s.replace("# HANDOVER - 2026-11-11 22:25", "# HANDOVER - 2026-11-11 23:05")
open(p, "w", encoding="utf-8").write(s)
print("ok")
