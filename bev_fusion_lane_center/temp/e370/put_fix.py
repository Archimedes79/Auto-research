p = "experiments/E370_e365-rule-distilled-init/experiment.md"
s = open(p, encoding="utf-8").read()
a = "## 3. Summary\n(empty until the runs exist)"
assert a in s
txt = ("## 3. Summary\n"
       "**Stage A (2026-11-17 04:22 - 04:38, runs `E370.a_s0 / a_s1`, 3000 steps, teacher rule, no crash, no resume; 0.25 s per step: the rule targets on the CPU per batch) and its FIXTURE (runs `E370.fixa_s0 / fixa_s1`, the 291-frame TRAIN sample, `jtoken_eval.py fixture --run e370_a_s<S> --split train --every 3`; files runs/e370_a_s<S>/token_fixture_train_every3_20261117T035820Z.json): PASS on both seeds.**\n"
       "| TRAIN sample, 291 frames | short links present (>= 0.95) | long links present (>= 0.80) | extra links (<= 0.05) | kept nodes agree (>= 0.99) | topo F1 head / c-rule (in gate) | rule links outside the gate | GT loss at stage-B step 0 (the stage-B kill reference) |\n"
       "|---|---|---|---|---|---|---|---|\n"
       "| seed 0 | 0.990 | 0.821 | 0.033 | 0.999 | 0.6441 / 0.6398 | 4 of 48 207 | 2.718 |\n"
       "| seed 1 | 0.990 | 0.827 | 0.031 | 0.999 | 0.6349 / 0.6381 | 11 of 47 935 | 2.804 |\n"
       "The stage-A head reproduces the rule per class (the long links of `lo_near`, the hard part, at 0.82 / 0.83), with a few extra links; on the TRAIN sample its topo F1 is within 0.005 of c-rule (+0.004 / -0.003), F1@1.0 0.827 / 0.825 against the rule's 0.811 / 0.807. The stage-B kill reference (the stage-A head's GT loss in eval mode on these frames) is 2.718 / 2.804, written here before stage B. **Stage B** waits for the weekday GPU window (the lane lead, 04:32); the stage-A VAL read-out (B-ADDS) is queued behind E373.\n\n(stage B follows)")
s = s.replace(a, txt)
open(p, "w", encoding="utf-8").write(s)
print("ok")
