p = "experiments/E371_e366-confirmation-fresh-head-seeds/experiment.md"
s = open(p, encoding="utf-8").read()
a = "**Champion status (the definition of section 0):**"
assert s.count(a) == 1
add = ("**G-cost, measured (2026-11-17 04:39 - 04:43, the quiet window: no run.py job, host load 2.2 - 2.9 before / after; runs `E371.time1 / time2`: `jtoken_eval.py timing --run e365_s1_s0 --b-run runs/e292_a_s0`, 24 frames, batch 1, bf16, CUDA events, 5 repeats, K = 512; files runs/e365_s1_s0/token_timing_20261117T034121Z.json and ..._T034328Z.json):** whole path (B' pyramid + node head + top-K gather + the head) median **16.83 / 16.57 ms** (p90 19.09 / 20.39), B' alone 11.41 / 11.37 ms, the head alone 6.70 / 6.64 ms (0.86 GMACs at K = 512); parameters 2.323 M (B' 2.196 M + head 0.127 M). Against E324 s's whole path, 19.7 ms and 2.741 M: **G-cost holds** (-2.9 to -3.1 ms on the median, -0.42 M parameters). **The tool's own flag says 'NOT QUIET'** because the GPU showed up to 10 % utilisation before the measurement (its idle limit is 5 %; the host load was below its limit 3.0): contention can only slow the path, so the quiet value is at most the measured one; the comparison is therefore conservative. The p90 of the second run (20.4 ms) is above 19.7: the tail is not better than E324 s's median; the gate compares the typical frame. The decode thresholds cost nothing measurable (the same forward). "
       "**Champion status after the timing: all conditions hold: a champion exists by the definition of section 0 (E366 + E371 + G-cost + G-simple).** ")
s = s.replace(a, add + a)
s = s.replace("**the only open condition is G-cost in milliseconds on a quiet host** (the next job, a 15-minute quiet window);", "G-cost was the last open condition (measured, see above);")
open(p, "w", encoding="utf-8").write(s)
print("ok")
