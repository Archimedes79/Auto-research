import os
p = os.path.join(os.environ["BFLC_ROOT"], "experiments", "E379_ensemble-distilled-head", "experiment.md")
s = open(p, encoding="utf-8").read()
a = "- **E2E:** yes: a loss term on the head's output;"
assert s.count(a) == 1
s = s.replace(a, "- **The teachers' own GT loss on the student's first TRAIN batches (MEASURED by Fable's coder, fp32 eval, head-seed order, bs 8, none weight 0.25; 2026-11-17 ~21:25):** seed 0: e373_cb_s0 first batch 0.4128, mean of 20 batches 0.3702 (single batches 0.240 - 0.495), e367_b25_s0 0.4164 / 0.3734 (0.232 - 0.504); seed 1: e373_cb_s1 0.3813 / 0.3524 (0.242 - 0.549), e367_b25_s1 0.3905 / 0.3553 (0.235 - 0.548): all four 20-batch means lie inside 0.30 - 0.50 (the start check), single batches do not, as the critic said.\n" + a)
open(p, "w", encoding="utf-8").write(s)
print("ok")
