import io
p = "experiments/E348_e324s-recipe-without-paint-branch/experiment.md"
s = io.open(p, encoding="utf-8").read()
a = "bit-identity with an uninterrupted run is not claimed (the data order after a resume is not the order of an uninterrupted run; the same deviation was named for `E317.g_s0` and `E318.u_s1`)"
b = "bit-identity with an uninterrupted run is not claimed (whether the data order and the random state after a resume equal those of an uninterrupted run was not checked; resumes were named as deviations for `E317.g_s0` and `E318.u_s1` too)"
assert s.count(a) == 1
io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
