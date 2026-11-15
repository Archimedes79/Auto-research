P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()
old = "one draw per sample, the same for all its frames; training mode only)"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "one draw per CLIP, shared by its key frames and all frames of a sample, the pixel noise differs per frame; seeded from (seed, step, micro-batch) so a resume repeats it; training mode only)")
open(P, "w").write(s)
print("ok")
