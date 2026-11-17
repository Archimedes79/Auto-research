import io
p = "experiments/E358_wiring-head-on-attached-ends/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
a = "## 6. Summary"
assert s.count(a) == 1
txt = ("**Attempt (4), the faithful two-loader live check (2026-11-16 12:46 - 12:49, run.py `E358.live_faithful_s0` / `E358.live_faithful_s1`, the command and the code as pre-stated): ACCEPTED on both seeds, 'all fields equal' in all 24 frames** (not even the one tolerated pair was used). The four attempts in order: (1) CPU as specified: REFUSED; (2) GPU as specified: REFUSED, frame 46 `pairs` by 0.0022; (3) GPU at the pair dump's batch size only: REFUSED, all frames' B graph; (4) GPU replaying how BOTH dumps were made: ACCEPTED. Each change was fixed before its run with the evidence of the previous one; (4) was allowed exactly once by the lead and pre-stated in this card before the code existed. It certifies the record construction and the join of the two models by row; it does NOT certify the TRAIN pass (single-loader, the pair model at batch size 4; up to 0.0022 in one pair of one frame on seed 0). **Released: `E358.w_s0` / `E358.w_s1` (`temp/e358/queue_w2.sh`, `--after` both faithful checks), then the scoring jobs `E358.sc_s0` / `E358.sc_s1`.**" + nl + nl)
s = s.replace(a, txt + a)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
