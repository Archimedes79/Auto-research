import io
p = "experiments/E374_s3-cross-attention-to-the-sd-map/experiment.md"
s = io.open(p, encoding="utf-8").read()
s = s.rstrip("\n") + """

**Launch gate result (2026-11-17 14:10; `run.py --no-card` diagnostics `E374.gate_m_s0` / `E374.gate_z_s0`, seed 0, 50 steps, bs 8, real head with Fable's hook 27eb572f, GPU idle, seconds per step over steps 11 - 50):** arm m **0.155 s/step**, peak GPU allocated 2.29 GB (reserved 2.59 GB), arm z **0.099 s/step**, 2.12 GB; valid tokens in the largest frame 512, no error. The block and the map build cost +57 % per step (the maps are built by the loader workers at 12 - 18 ms per frame; the step is host-bound), +0.17 GB. At 12000 steps: m about 31 min, z about 20 min solo; two runs in parallel about 1.3 times slower: `--timeout 90` (minutes) for every run. The kill measures after 50 steps (untrained): v (other-sample map) 0.047, v_none 0.145. Run order when a slot is free: z_s0, m_s0, z_s1, m_s1, z2_s0 (one after another, about 2.0 h).
"""
io.open(p, "w", encoding="utf-8").write(s)
