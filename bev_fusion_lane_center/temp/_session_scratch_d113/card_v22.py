import io
p = "experiments/E368_s2-cross-attention-to-the-past/experiment.md"
s = io.open(p, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep("u = RMS(block(x, past) - block(x, none key only)) / RMS(x) on the frames that have a past, averaged over the last 500 steps, is below 0.01 on BOTH seeds of arm r (the verdict is then TIE with the status 'the block did not learn to read the past'; the block output alone is no measure: it contains the constant W_o v_none)",
    "u = RMS(block(x, past) - block(x, the past of ANOTHER SAMPLE of the batch)) / RMS(x) on the frames that have a past, averaged over the last 500 steps, is below 0.01 on BOTH seeds of arm r (the verdict is then TIE with the status 'the block did not learn to read the past'; the none-only version `u_none` is logged as a diagnostic: a content-free attention satisfies it, W_o (mean v_past - v_none) is a constant; v2.1 of 2026-11-17 07:15 after the second critic pass found this on E374)")
rep("and u of the kill per stream position.", "and u / u_none of the kill per stream position (the stream read-out has one sample: its donor is the `shuffled` mode).")
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
