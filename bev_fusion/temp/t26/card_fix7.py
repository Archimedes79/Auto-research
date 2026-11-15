"""scratch: the CRN run ids join the base arm (append only: the reference runs are registered like the base-side ones)"""
P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()
old = '"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base", "t26.base_eval_vel"], "cards"'
assert s.count(old) == 1
s = s.replace(old, '"run_ids": ["t26.base_eval", "t26.base_eval43", "t26.ref_maps_base", "t26.base_eval_vel", "t26.ref_crn_smoke", "t26.ref_crn"], "cards"')
old2 = "(the three base ids are registered in the card's `base` arm"
if old2 in s:
    s = s.replace(old2, "(the base-side ids and the two CRN ids are registered in the card's `base` arm")
open(P, "w").write(s)
print("ok")
