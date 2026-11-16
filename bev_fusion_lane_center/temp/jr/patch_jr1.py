def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))

b = "build_junction_report.py"
sub(b, 'RULE = J("runs/e287/rule_control.json")', '''import glob as _glob
RULE = {}
for _p in sorted(_glob.glob(os.path.join(ROOT, "runs/e287/rule_control_thr0.5_*.json"))):                           # strict one-to-one matching, keypoint threshold 0.5 (critic fix of 2026-11-12)
    RULE.update(json.load(open(_p)))''')
sub(b, 'SPLIT = {s: (J(f"runs/e300/splits_e284_h2_s{s}.json") or {}).get(f"e284_h2_s{s}") for s in (0, 1)}', 'SPLIT = {s: (J("runs/e300/splits_thr0.5_e284_h2_s0_e284_h2_s1.json") or {}).get(f"e284_h2_s{s}") for s in (0, 1)}')
sub(b, '''            rows.append([f"rule control on the keypoints of {run}", "-", "-", f3(v["rule"]["recall"]), f3(v["rule"]["precision"]), "-", "-", "-", "-"])''', '''            rows.append([f"rule control on the keypoints of {run} (ceiling {f3(v['ceiling_both_ends_matched'])})", "-", "-", f3(v["rule"]["recall"]), f3(v["rule"]["precision"]), "-", "-", "-", "-"])''')
sub(b, '"E284 end head: 0.29 / 0.29; rule control 0.37 (precision 0.13)"', '"E284 end head: 0.29 / 0.29 (precision 0.74); rule control 0.36 (precision 0.34); ceiling of the keypoints 0.43"')
t = "report_text.py"
sub(t, "Reading of the controls: the rule delivers five to six times as many pairs and is wrong for 86 % of them (precision 0.13 against the head's 0.59), but its recall (0.37) is the ceiling that the E284 keypoints allow: two ends are found for about 0.34 - 0.38 of the true pairs. ",
    "Reading of the controls (strict one-to-one matching, the same keypoints for the rule and the head; the first version of the controls was too loose and was corrected after a critic pass): the rule delivers about three times as many pairs and is wrong for two thirds of them (precision 0.33 against the head's 0.74), but its recall (0.35) is higher than the head's (0.29) and the ceiling of the E284 keypoints is 0.42 - 0.43 (both ends of a pair found). ")
sub(t, "the rule control reaches 0.37 on E284's keypoints.", "the keypoints of E284 allow 0.43 (both ends found), the head delivers 0.29, the rule 0.36 at precision 0.34.")
print("ok")
