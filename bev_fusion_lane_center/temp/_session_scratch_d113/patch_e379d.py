import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_ensemble-distilled-head", "experiment.md")
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep("and the teachers' own GT loss on the first batch, run in fp32 eval, lies in 0.30 - 0.50 (MEASURED on the 291-frame TRAIN sample at none weight 0.25: A 0.3728, B 0.3716; logged final batch losses 0.36 - 0.39;", "and the MEAN of the teachers' own GT loss over the FIRST 20 batches, computed at start in fp32 eval, lies in 0.30 - 0.50 (MEASURED on the 291-frame TRAIN sample at none weight 0.25: A 0.3728, B 0.3716; logged final batch losses 0.36 - 0.39; Fable's code critic: single batches of the teachers' last 1000 training steps fall outside 0.30 - 0.50 in 24 - 26 % of cases, p0.5 - p99.5 about 0.19 - 0.62, SD about 0.08, so a SINGLE-batch check would refuse one start in four; the 20-batch mean has SE about 0.018;")
rep("Logged: the KD parts every logged step and at step 1, the teachers' own GT loss on the first batch.", "Logged: the KD parts every logged step and at step 1, the teachers' own GT loss on the first batch and its mean over the first 20 batches (the start check).")
open(p, "w", encoding="utf-8").write(s)
print("ok")
