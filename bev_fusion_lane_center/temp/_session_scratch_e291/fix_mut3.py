p = "temp/e291/mutate.py"
s = open(p, encoding="utf-8").read()
old = '''    s = open(f"{d}/{f}", encoding="utf-8").read()
    s2, n = re.subn(pat, lambda _m: rep, s)
    assert n == 1, f"{mid}: the pattern matched {n} times"
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s2)'''
new = '''    s2 = open(f"{d}/{f}", encoding="utf-8").read()
    for pt, rp in zip(pat if isinstance(pat, list) else [pat], rep if isinstance(rep, list) else [rep]):               # a mutation of two guards at once: lists
        s2, n = re.subn(pt, lambda _m, rp=rp: rp, s2)
        assert n == 1, f"{mid}: the pattern matched {n} times"
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s2)'''
assert old in s
s = s.replace(old, new)
old2 = '    ("p12_junction_ignores_token"'
new2 = '''    ("p11b_no_detach_no_nograd", JPR, [r"@torch\.no_grad\(\)\ndef features", r"features\(pos\.detach\(\)\.float\(\), D\.detach\(\)\.float\(\)"], ["def features", "features(pos.float(), D.float()"], JS, "the prior path reaches the keypoint positions (both guards off: p11 alone is equivalent, either guard suffices)"),
    ("p12_junction_ignores_token"'''
assert old2 in s
s = s.replace(old2, new2, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
