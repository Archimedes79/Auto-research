p = "plan.md"
s = open(p, encoding="utf-8").read()
old = "rerun on the corrected data, with velocity error on box interiors vs edges and per box (mean v)\n"
new = old + ("    corrected data, best setup, seed 0: 2 / 5 - velocity error 1.64 m/s per cell but 0.94 per box (the box's mean velocity): the floor is per-cell "
             "scatter, not bias; interiors 1.54 vs edges 1.68 (not the splat blur); pedestrians 0.95; LANES 3.37 m/s (the aperture problem, now measured - the "
             "simulated radar, like a real one, does not see lane paint, so Doppler cannot fix it); raw move 0.725 (GT 0.681), extrapolation 0.524 = 82 % of "
             "GT 0.641, decoder share 0.145 (lanes make it harder). Queued: zero-motion prior 0.03 (lanes), raw-warp 3.0 (decoder share)\n")
assert s.count(old) == 1
open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
print("ok")
