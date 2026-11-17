import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 6. Summary"
assert s.count(a) == 1
txt = ("**Live check (CPU, 2026-11-16 11:10, as written in the rule above: `--device cpu`, `temp/e358/live_check.sh`): REFUSED on both seeds** (seed 0: 9 of 24 frames differ, `ends` / `attrs` mostly, 3 with `pairs`; seed 1 likewise). **Control (11:14, `temp/e358/live_control.sh`): the same check on CPU for E312's keypoint records (E325's accepted path) is REFUSED on the same frames (40, 41, 48, 56, 58, 59, 61, 62, `ends` / `attrs`)**: the differences are B's delivered graph of a CPU live pass against the GPU numerics of the stored dumps, not the attached ends (E325's own acceptance, 'one pair near p = 0.1 in one frame', was a GPU pass). By the written rule REFUSED means no GPU training job, no change of the tolerance, and the lead decides; **the lead was asked at 11:15 whether to repeat the SAME check (same fields, same tolerance) on the GPU, the device of the stored dumps, as a short `run.py --gpu` job; nothing was changed in the meantime.**" + nl + nl)
s = s.replace(a, txt + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
