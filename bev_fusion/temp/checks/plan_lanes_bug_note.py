"""one-off: record the box-test lane data bug in plan.md 7.3 and tests/README.md"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "keep the decoder from carrying motion as training goes on (raw-warp weight, or the decoder on the raw move)\n"
new = old + ("    DATA BUG found (2026-10-26): with 2x GT supersampling the 0.15 m lane lines fell exactly between the sample points - F0 of the box test had NO lanes "
             "(F1 / F2 only dashed pieces after the ego rotation: the dashed lines in all box pictures), and the lane mask (> 0.3) was empty too, so 'EPE lanes "
             "0.000' was never a measurement. All box-test numbers above are vehicles-only scenes in F0; static structure was not tested. Fixed (4x, lane mask "
             "> 0.05); the best setup (2 seeds) and the baseline rerun on the corrected data, with velocity error on box interiors vs edges and per box (mean v)\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))

p = "tests/README.md"
s = open(p, encoding="utf-8").read()
old = "only '< 1 m/s' fails (the raw move 0.77 > GT 0.69 shows some gaming on the raw map) |"
new = ("only '< 1 m/s' fails (the raw move 0.77 > GT 0.69 shows some gaming on the raw map); these runs had a DATA BUG - no lane lines in F0 (2x "
       "supersampling missed them; fixed, 4x) - rerun on the corrected data |")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
