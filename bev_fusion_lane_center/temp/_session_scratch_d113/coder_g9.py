def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, s.count(a), a[:80])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)


edit("training/README.md", [("`u` (the kill measure) is logged", "`u` (the kill measure: the block's update with the past against its update with the past of ANOTHER sample of the batch, over RMS(x); the arm needs `--bs` >= 2) and `u_none` (against the none key only, no test of use) are logged")])
edit("training/grid/README.md", [("`UseMeter` (the kill measure u as a forward hook)", "`roll_past` + `PastBlock.use` + `UseMeter` (the kill measure u = RMS(update with the past - update with the past of ANOTHER sample of the batch) / RMS(x), a forward hook; a batch of one needs a donor past; `u_none` against the none key only is reported next to it: any content-free attention satisfies it)"),
                                 ("(hand-computed values, a loop reference of the block in float64,", "(hand-computed values, a loop reference of the block in float64, a block that ignores the past content (u = 0, u_none > 0) and one that reads it (u by hand),")])
edit("evaluation/README.md", [("`use` = u and the RMS of the carried state per stream position", "`use` = u (the update with the past against the update with ANOTHER scene's past, the donor of `donor_plan`; VAL) and u_none (against the none key only: no test of use) and the RMS of the carried state per stream position")])
print("ok")
