import io
import subprocess

p = "experiments/E353_paint-branch-width-25/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
assert "## 2b. Launch as it happened" not in s
txt = """## 2b. Launch as it happened (2026-11-15; named deviations)
- **Seed 0 only so far.** The chain `temp/day29/day29.sh` was started 17:29 with `DAY29_SKIP_WAIT=1 DAY29_STOP_AFTER_S0=1` (the GPU was free before E354, Sunday: no daytime limit, D100; the quiet-host timing of the original chain order could not run: jtiming refused twice because the desktop held the GPU at 6 - 60 % utilisation): the step-0 rows `E353.n1z_s0 / n1z_s1` (CPU, `--steps 0`: measured GMACs per frame 8.2016 -> 7.0957, saved 1.1059 = 13.5 %, both seeds), then `E353.cp_s0` and `E353.n1_s0` in the two free slots (17:32 - 18:49, 2.7 s per step on a host loaded by two training runs and the E354 preparation). Seed 1, the dumps, the cards and the rule replay are NOT started: they follow after E354's runs (the Fable session's recursive-fusion pair stages hold both slots until about 22:45); the quiet-host timing waits for a night gap.
- **`E353.cp_s0` was resumed:** at step 1010 the WSL 'CUDA error: unknown error' hit it three times (`_pyr.sh` retries 1 - 3), each time resuming from the step-1000 checkpoint; it finished at 18:49. **The data order of cp_s0 after step 1000 therefore differs from n1_s0's** (the resume re-seeds from the checkpoint step), so the paired comparison of cp against n1 on seed 0 carries that difference; the rule replay says so, and a cp_s0 with a clean run is the repair if the comparison against cp is decisive. `E353.n1_s0` ran clean (1500 steps, no retry).
- **Equal-budget caveat:** cp and n1 started in the same minute from the same init with the same seed; nothing else differs but the narrowing, except the resume of cp_s0 named above.
"""
anchor = "## 3. Summary"
assert s.count(anchor) == 1
s = s.replace(anchor, txt + "\n" + anchor)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
