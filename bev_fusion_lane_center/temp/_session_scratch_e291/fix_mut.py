p = "temp/e291/mutate.py"
s = open(p, encoding="utf-8").read()
rep = [
 ('r"lat_i\[:, None, :\] > lat_i\[:, :, None\]\)", "lat_i[:, None, :] < lat_i[:, :, None])"', 'r"lat_k > lat_i\[:, None, :\]\) \|", "lat_k < lat_i[:, None, :]) |"'),
 ('r"\(has & ok\)\[:, None, :\]", "has[:, None, :]"', 'r"& ok\[:, :, None\] & \(kind", "& (kind"'),
 ('r"found = same\.sum\(-1\)\.float\(\) ", "found = same.sum(-1).float() + 0.0"', None),
]
for old, new in rep[:2]:
    assert old in s, old
    s = s.replace(old, new)
a = s.index('    ("p18_real_ignored_in_found"')
b = s.index("\n", a)
s = s[:a] + '    ("p18_single_assignment", JPR, r"mem = elig\.gather\(2, own\) &", "mem = (own == arm[:, :, None]) &", JS, "the keypoints of an arm are only the ones assigned to it (a keypoint near two arms counts for one)"),' + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
