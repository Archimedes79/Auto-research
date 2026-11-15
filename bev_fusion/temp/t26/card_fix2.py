P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()
a = '"run_ids": ["t26.base_eval"]}},'
assert s.count(a) == 1
s = s.replace(a, '"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base"]}},')
b = "`t26.ref_maps` (maps scorer: our checkpoint and the cached GaussianCaR logits, the same frames)"
assert s.count(b) == 1
s = s.replace(b, "`t26.ref_maps_base` and `t26.ref_maps_one` (maps scorer: the base's and the t26 checkpoint, against the cached GaussianCaR logits, the same frames; the three base ids are registered in the card's `base` arm, the t26 ones are appended when they are launched)")
open(P, "w").write(s)
print("ok")
