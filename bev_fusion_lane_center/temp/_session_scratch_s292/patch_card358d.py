import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:80], s.count(a))
    s = s.replace(a, b)
rep("**Status: card v2, not yet passed by a second critic (section 4);", "**DEVIATION by the lead 2026-11-16: the live check was specified with --device cpu; the E312 control shows the CPU path cannot reproduce the stored GPU dumps, so the check is repeated on the device of the dumps (cuda), as a `run.py --gpu` job with its own ledger row, behind the two E361 runs of the Fable session; the tolerance, the fields, the 24 frames and the rule stay exactly as written, the CPU result (REFUSED on both seeds, 9 of 24 frames, the same frames as the E312 control) stays in the card as a recorded fact, no tolerance widened. ACCEPTED on the GPU: `E358.w_s0` / `w_s1` start at once; REFUSED on the GPU too: the card stops, no GPU run, no tolerance change, the loader question is written into the card and goes to the lead.** **Status: card v2, not yet passed by a second critic (section 4);")
rep("A code critic on the build (section 2 step 5) is required before any GPU job: **done 2026-11-16 10:30, no blocking defect (section 5).**", "A code critic on the build (section 2 step 5) is required before any GPU job: **done 2026-11-16 10:30, no blocking defect (section 5).** **The lead decided the device question of the live check on the evidence of the E312 control (11:20); the critic was not asked again, nothing but the device changed.**")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
