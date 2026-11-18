import io
p = "experiments/E368_s2-cross-attention-to-the-past/experiment.md"
s = io.open(p, encoding="utf-8").read()
s = s.rstrip("\n") + """

**Launch gate result (2026-11-17 08:37; `run.py --no-card` diagnostics `E368.gate_r_s0` / `E368.gate_z_s0`, seed 0, 50 steps, real head with Fable's hook, GPU idle, seconds per step over steps 11 - 50):** arm r **0.268 s/step**, peak GPU allocated 3.32 GB (reserved 3.43 GB); arm z **0.220 s/step**, 2.96 GB; both 4 sequences x 3 frames, valid tokens in the largest frame 512, no error. The block costs +22 % per step and +0.36 GB. At 8000 steps: r about 36 min, z about 29 min alone; two runs in parallel about 1.3 times slower: `--timeout 90` (minutes) for every run (pessimistic: x 1.3 x 1.3 on the parallel figure). The kill measures after 50 steps (untrained): u (other sample) 0.0042, u_none 0.102: **the none-only measure is already 24 times larger than the other-sample measure before any learning, which is the flaw the second critic named; the kill uses u.** The five runs: z_s0, r_s0 first (both slots), then z_s1, r_s1, z2_s0 as the slots free.
"""
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
