p = "temp/jr/report_text.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:80]
    s = s.replace(old, new, 1)


sub("the heading of the new keypoints is better (4.4 degrees median error 3.2 against 8.9 / 7.4), the position slightly worse (1.38 m against 1.18 m)", "the heading of the new keypoints is better (mean error 4.4 degrees against 8.9), the position slightly worse (mean error 1.38 m against 1.18 m)")
# NEXT and WEAK: the first item of NEXT and the first of WEAK
sub("<li>Read the night results by their written rules (E286 arms, E287 arms, the joint arm, the line probe) and refine every TIE (D82).</li>",
    "<li>Today's runs (day chain 3, both seeds, scored right after each run): E305 maps and pair head trained together, E302 degree budget, E307 narrower end targets, E298 driven weight, E306 exact polygon targets, E303 handle scales, E304 far range; read them by the written rules and refine every TIE (D82). The pair decision is the first target (same-ends comparison).</li>")
sub("<li><b>Pair recall is capped by the end recall</b>;", "<li><b>The pair decision is weaker than the E284 head on the same ends</b> (AUC 0.94 against 0.975, recall at equal precision half as high); E305 / E302 / E309 are the answers under test.</li>"
    "\"\n        \"<li><b>Pair recall is capped by the end recall</b>;")
open(p, "w", encoding="utf-8").write(s)
print("ok")
