def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))
t = "report_text.py"
sub(t, "<li><b>What might be overlooked:</b> occlusion matters little for the ends (2 points), the range matters a lot; left- and right-hand traffic differ (Boston 0.40, Singapore 0.29 pair recall for E284);",
    "<li><b>What might be overlooked:</b> occlusion does not matter for the ends (0.57 visible, 0.58 hidden), the range matters a lot and so does the paint (painted entries are found 8 - 12 points more often); left- and right-hand traffic differ (Boston 0.35, Singapore 0.24 pair recall for E284);")
sub(t, "<li><b>Honest negatives:</b>", '''<li><b>Corrected after an independent code review (2026-11-11):</b> (1) the rule control and the failure splits counted a GT end as found when ANY keypoint lay near it, which flattered the recall: with strict one-to-one matching the rule's precision is 0.34 (not 0.13), the "more lanes are easier" effect disappears, paint starts to matter; (2) exits were marked "painted" by the stop line of their own arm's entry side: now crossings only (18.5 % of the exits, not 28 %); (3) the rank features of the pair head used the injected ground-truth keypoints in training but only real peaks at evaluation: now real peaks in both; (4) the legacy polygon fill puts every zone / paint target half a cell off the road (36 % wrong cells): an exact mode exists (E306), the numbers of this page are on the legacy targets and say so; (5) both maps were compared at the same fixed keypoint threshold although they are calibrated differently: the scorer now also reports the best threshold per map.</li>
        <li><b>Honest negatives:</b>''')
print("ok")
