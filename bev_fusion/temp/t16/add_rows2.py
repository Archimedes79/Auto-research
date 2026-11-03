"""scratch: HEAD copies of three docs with the rows of the split gradient clip (each insertion must match exactly once)"""
import subprocess

T = "bev_fusion/temp/t16/"


def head(path):
    return subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True, check=True, encoding="utf-8").stdout


# tests/README.md: a row after the grad-scaler row
s = head("bev_fusion/tests/README.md")
lines = s.split("\n")
idx = [i for i, ln in enumerate(lines) if ln.startswith("| `test_grad_scaler.py` |")]
assert len(idx) == 1
row = ("| `test_clip_split.py` | the split gradient clip of `training/train.py` (`TrainConfig.grad_clip_split`, `clip_groups`, `clip_gradients`; CPU, seconds, toy modules and the REAL `train()` on a toy): "
       "no split = the one global torch call bit for bit; with a split each listed module and the rest are clipped on their own (a block with a 100 x larger gradient does not scale the others down: "
       "the t16 finding, the motion block 36 - 131 against the others' 0.1 - 41); the returned norm is the total before the clip; NaN / inf in any group is no step; groups without gradients, "
       "two prefixes, overlaps and a bare string, the fp16 scaler's overflow, the resume refusal of an old checkpoint, the weights after a step against the arithmetic, the call site in `train()`, "
       "a typo failing before the data are built | each listed module clipped to the limit, the rest unscaled | passes; the 14 mutants of the code critic die (the two that survived the first version, an ignored split in optimizer_step and a taken-set from the first group, and the dropped call site and the resume hole are covered) |")
lines.insert(idx[0] + 1, row)
open(T + "tests_README.md", "w", encoding="utf-8").write("\n".join(lines))

# training/README.md: the train.py row names the new option
s = head("bev_fusion/training/README.md")
lines = s.split("\n")
idx = [i for i, ln in enumerate(lines) if ln.startswith("| `train.py` |")]
assert len(idx) == 1
parts = lines[idx[0]].split(" | ")
parts[1] += " `grad_clip_split` (opt-in: module prefixes whose gradients are clipped on their own, the rest as one group; the motion block's gradient is 30 x the others')."
lines[idx[0]] = " | ".join(parts)
open(T + "training_README.md", "w", encoding="utf-8").write("\n".join(lines))

# run_gates.sh: the gate after the grad scaler
s = head("bev_fusion/tests/run_gates.sh")
a = 'run "grad scaler"       "$PY" tests/test_grad_scaler.py            # amp fp16: scaler exact against plain steps, overflow, bad-step rule, checkpoint\n'
assert s.count(a) == 1
b = a + 'run "clip split"        "$PY" tests/test_clip_split.py             # grad_clip_split: each listed module and the rest clipped on their own, the global call unchanged, resume, call site\n'
open(T + "run_gates.sh", "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
print("rows written")
