p = "experiments/E366_e365-decode-selected-on-train/experiment.md"
s = open(p, encoding="utf-8").read()
a = "## 3. Summary\n(empty until the runs exist)"
assert a in s
s = s.replace(a, "## 3. Summary\n**The chosen cell (written 2026-11-17 02:47 system clock, BEFORE any VAL score of a variant; by the rule of section 0 on the two TRAIN-sample sweep tables `runs/e365_s1_s<S>/token_sweep_train_every3.json`, 291 frames each, runs `E366.sweep_s0 / s1`, scratch `temp/e366/choose.py`): link threshold 0.1, keep threshold 0.6, none gate OFF** (mean TRAIN-sample topo F1 0.7222 (0.7200 / 0.7245), mean dup_frac 0.0435 <= 0.05; precision 0.736 / 0.728, recall 0.705 / 0.721, lanes found 0.780 / 0.785, F1@1.0 0.900 / 0.901; the record cell on the same frames: 0.6228 (0.6191 / 0.6264), precision 0.865 / 0.867, recall 0.482 / 0.490). As said before the sweep, the rule took a loose cell, but not the loosest: (0.05, *, off) and (0.1, 0.7, off) break the duplicate guard (0.079 - 0.106 and 0.0501). This cell is applied ONCE to VAL now (runs `E366.val_s0 / val_s1`).\n\n(result follows)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
