import io
p = "experiments/E368_s2-cross-attention-to-the-past/experiment.md"
s = io.open(p, encoding="utf-8").read()
a = "(the stream read-out has one sample: its donor is the `shuffled` mode)."
assert s.count(a) == 1
b = "(the stream read-out has one sample: its donor is the state of another scene at the same step index, warped by this frame's pose, so its `u` is not the same comparison as the training one: **THE KILL MEASURE IS THE TRAINING `u`** (the past of another sample of the batch in its own lane frame, the last 500 steps = five logged 100-step windows); the read-out `u` is reported by stream position as a diagnostic, never a kill)."
io.open(p, "w", encoding="utf-8").write(s.replace(a, b))
print("ok")
