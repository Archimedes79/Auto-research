import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "**PRE-STATEMENT of attempt (4)"
assert s.count(a) == 1
txt = ("**Diagnostic of attempt (2), no verdict (`E358.livediag_s0`, run.py --gpu --no-card, 12:19): in the GPU live pass at batch size 4 (seed 0) 23 of 24 frames are equal in every field; frame 46 differs in `pairs` only: the largest absolute difference is 0.0022 (tolerance 0.002) and 2 of its 8 pair rows lie above 0.002**, i.e. numerics of one attached end, not a switch of a candidate by metres. The diagnostic does not change the rule: the acceptance stays 'every field equal except at most ONE pair near p = 0.1 in ONE frame'." + nl + nl)
s = s.replace(a, txt + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
