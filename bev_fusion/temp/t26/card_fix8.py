"""scratch: the front-sector scoring runs join the base arm (append only)"""
P = "/mnt/c/Auto-research/bev_fusion/experiments/t26_one_long_run/experiment.md"
s = open(P).read()
old = '"t26.ref_crn_smoke", "t26.ref_crn"], "cards"'
assert s.count(old) == 1
s = s.replace(old, '"t26.ref_crn_smoke", "t26.ref_crn", "t26.ref_front_base", "t26.ref_front_one"], "cards"')
open(P, "w").write(s)
print("ok")
